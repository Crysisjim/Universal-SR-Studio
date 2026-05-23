#!/usr/bin/env python3
"""
neo_sr_slave.py — Minimal Distributed Training Slave for Universal SR Studio
=============================================================================
Lightweight TCP server (zero extra dependencies beyond torch/yaml/psutil).
Runs on worker PCs in the LAN, receives commands from the Master GUI.

Usage:
    python neo_sr_slave.py                     # default port 5001
    python neo_sr_slave.py --port 5002         # custom port
    python neo_sr_slave.py --name "PC-Gaming"  # custom display name

Protocol: TCP + JSON lines (one JSON object per line, \n terminated)
Commands: ping, gpu_info, benchmark, sync_dataset, start_training, stop_training, status, shutdown
"""

import argparse
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ─── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("slave")

# ─── Constants ────────────────────────────────────────────────
VERSION = "1.0.0"
CACHE_DIR = Path(os.environ.get("SR_SLAVE_CACHE", Path.home() / "sr_slave_cache"))
DATASET_DIR = CACHE_DIR / "datasets"
CONFIG_DIR = CACHE_DIR / "configs"


# ─── GPU Detection ───────────────────────────────────────────
def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU capabilities using torch and pynvml/nvidia-smi."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "torch_version": "",
        "cuda_version": "",
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda or ""
        info["gpu_count"] = torch.cuda.device_count()

        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_mem // (1024 * 1024)
            gpu = {
                "index": i,
                "name": props.name,
                "total_vram_mb": total_mb,
                "compute_capability": f"{props.major}.{props.minor}",
                "supports_amp": props.major >= 7,
                "supports_bf16": props.major >= 8,
            }
            # Try to get current memory usage
            try:
                free, total = torch.cuda.mem_get_info(i)
                gpu["free_vram_mb"] = free // (1024 * 1024)
            except Exception:
                gpu["free_vram_mb"] = total_mb

            info["gpus"].append(gpu)
    except ImportError:
        log.warning("PyTorch not installed — GPU info unavailable")
    except Exception as e:
        log.error(f"GPU detection error: {e}")

    return info


def run_benchmark(seconds: int = 10) -> Dict[str, Any]:
    """Run a quick GPU benchmark to measure relative performance."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"error": "No CUDA GPU available"}

        results = []
        for gpu_idx in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{gpu_idx}")
            torch.cuda.set_device(device)

            # Warm up
            x = torch.randn(64, 3, 64, 64, device=device)
            w = torch.randn(64, 3, 3, 3, device=device)
            for _ in range(5):
                torch.nn.functional.conv2d(x, w, padding=1)
            torch.cuda.synchronize()

            # Benchmark: measure conv2d throughput
            start = time.perf_counter()
            ops = 0
            while time.perf_counter() - start < seconds:
                torch.nn.functional.conv2d(x, w, padding=1)
                ops += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            results.append({
                "gpu_index": gpu_idx,
                "ops_per_second": ops / elapsed,
                "total_ops": ops,
                "duration": round(elapsed, 2),
            })

        return {"benchmarks": results}
    except Exception as e:
        return {"error": str(e)}


# ─── Dataset Sync ────────────────────────────────────────────
def sync_dataset(source_path: str, name: str, callback=None) -> Tuple[bool, str]:
    """
    Sync dataset from network share to local cache.
    Uses robocopy on Windows, rsync on Linux.
    """
    dest = DATASET_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(source_path):
        return False, f"Source not found: {source_path}"

    if sys.platform == "win32":
        cmd = ["robocopy", source_path, str(dest), "/MIR", "/MT:8", "/NJH", "/NJS", "/NDL", "/NFL"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, creationflags=0x08000000)
            for line in proc.stdout:
                line = line.strip()
                if line and callback:
                    callback(f"[sync] {line}")
            proc.wait()
            # robocopy returns 0-7 for success
            if proc.returncode <= 7:
                return True, f"Synced to {dest}"
            return False, f"robocopy error code {proc.returncode}"
        except Exception as e:
            return False, str(e)
    else:
        cmd = ["rsync", "-av", "--progress", f"{source_path}/", str(dest)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in proc.stdout:
                if callback:
                    callback(f"[sync] {line.strip()}")
            proc.wait()
            return proc.returncode == 0, f"Synced to {dest}"
        except Exception as e:
            return False, str(e)


# ─── Training Process ────────────────────────────────────────
class TrainingManager:
    """Manages a single training subprocess (torchrun)."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.log_lines: list = []
        self.config_path: Optional[str] = None
        self._log_lock = threading.Lock()

    def start(self, config_path: str, master_addr: str, master_port: int,
              node_rank: int, nnodes: int, nproc_per_node: int,
              engine: str = "neosr", python_path: str = "python") -> Tuple[bool, str]:
        """Launch torchrun for distributed training."""
        if self.is_running:
            return False, "Training already running"

        self.config_path = config_path
        self.log_lines.clear()

        # Determine training script
        if engine.lower() == "neosr":
            train_module = "-m neosr.train"
        else:
            train_module = "-m traiNNer.train"

        cmd = [
            python_path, "-m", "torch.distributed.run",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            train_module,
            "-opt", config_path,
        ]

        log.info(f"Starting training: {' '.join(cmd)}")

        try:
            creationflags = 0x08000000 if sys.platform == "win32" else 0
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=creationflags,
            )
            self.is_running = True

            # Log reader thread
            t = threading.Thread(target=self._read_logs, daemon=True)
            t.start()

            return True, f"Training started (PID: {self.process.pid})"
        except Exception as e:
            return False, str(e)

    def stop(self) -> Tuple[bool, str]:
        """Stop current training gracefully."""
        if not self.is_running or not self.process:
            return False, "No training running"

        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGINT)

            self.process.wait(timeout=15)
            self.is_running = False
            return True, "Training stopped"
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.is_running = False
            return True, "Training force-killed"
        except Exception as e:
            return False, str(e)

    def get_status(self) -> Dict[str, Any]:
        """Get training status."""
        if self.process and self.process.poll() is not None:
            self.is_running = False

        with self._log_lock:
            recent = self.log_lines[-20:] if self.log_lines else []

        return {
            "running": self.is_running,
            "pid": self.process.pid if self.process and self.is_running else None,
            "config": self.config_path,
            "log_count": len(self.log_lines),
            "recent_logs": recent,
        }

    def _read_logs(self):
        """Background thread: read process stdout."""
        try:
            for line in self.process.stdout:
                line = line.rstrip()
                with self._log_lock:
                    self.log_lines.append(line)
                    # Keep last 5000 lines
                    if len(self.log_lines) > 5000:
                        self.log_lines = self.log_lines[-2500:]
        except Exception:
            pass
        finally:
            self.is_running = False


# ─── TCP Server ──────────────────────────────────────────────
class SlaveServer:
    """TCP JSON-lines server for receiving Master commands."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5001, name: str = ""):
        self.host = host
        self.port = port
        self.name = name or platform.node()
        self.trainer = TrainingManager()
        self.running = True
        self.server_socket: Optional[socket.socket] = None

    def start(self):
        """Start listening for connections."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(2.0)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        log.info(f"═══════════════════════════════════════════")
        log.info(f"  SR Slave v{VERSION} — {self.name}")
        log.info(f"  Listening on {self.host}:{self.port}")
        log.info(f"  Cache: {CACHE_DIR}")
        log.info(f"═══════════════════════════════════════════")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                log.info(f"Connection from {addr}")
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self):
        """Shutdown server."""
        self.running = False
        self.trainer.stop()
        if self.server_socket:
            self.server_socket.close()

    def _handle_client(self, conn: socket.socket, addr: tuple):
        """Handle a single client connection (one command per line)."""
        conn.settimeout(30.0)
        buf = ""
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        request = json.loads(line)
                        response = self._process_command(request)
                    except json.JSONDecodeError:
                        response = {"ok": False, "error": "Invalid JSON"}

                    resp_bytes = (json.dumps(response) + "\n").encode("utf-8")
                    conn.sendall(resp_bytes)
        except (ConnectionResetError, BrokenPipeError, socket.timeout):
            pass
        except Exception as e:
            log.error(f"Client error: {e}")
        finally:
            conn.close()
            log.info(f"Disconnected: {addr}")

    def _process_command(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a command and return response."""
        cmd = req.get("cmd", "").lower()
        data = req.get("data", {})

        if cmd == "ping":
            return {
                "ok": True, "name": self.name, "version": VERSION,
                "platform": platform.system(), "hostname": platform.node(),
            }

        elif cmd == "gpu_info":
            return {"ok": True, **get_gpu_info()}

        elif cmd == "benchmark":
            duration = data.get("seconds", 10)
            return {"ok": True, **run_benchmark(duration)}

        elif cmd == "sync_dataset":
            source = data.get("source", "")
            name = data.get("name", "dataset")
            ok, msg = sync_dataset(source, name)
            return {"ok": ok, "message": msg}

        elif cmd == "start_training":
            # Save config to local file
            config_content = data.get("config_content", "")
            config_name = data.get("config_name", "train.yml")
            config_path = CONFIG_DIR / config_name

            if config_content:
                config_path.write_text(config_content, encoding="utf-8")

            ok, msg = self.trainer.start(
                config_path=str(config_path),
                master_addr=data.get("master_addr", "127.0.0.1"),
                master_port=data.get("master_port", 29500),
                node_rank=data.get("node_rank", 0),
                nnodes=data.get("nnodes", 1),
                nproc_per_node=data.get("nproc_per_node", 1),
                engine=data.get("engine", "neosr"),
                python_path=data.get("python_path", "python"),
            )
            return {"ok": ok, "message": msg}

        elif cmd == "stop_training":
            ok, msg = self.trainer.stop()
            return {"ok": ok, "message": msg}

        elif cmd == "status":
            return {"ok": True, **self.trainer.get_status()}

        elif cmd == "get_logs":
            count = data.get("count", 50)
            with self.trainer._log_lock:
                lines = self.trainer.log_lines[-count:]
            return {"ok": True, "logs": lines}

        elif cmd == "shutdown":
            log.info("Shutdown requested by master")
            self.stop()
            return {"ok": True, "message": "Shutting down"}

        else:
            return {"ok": False, "error": f"Unknown command: {cmd}"}


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SR Slave — Distributed Training Worker")
    parser.add_argument("--port", type=int, default=5001, help="TCP port (default: 5001)")
    parser.add_argument("--name", type=str, default="", help="Display name for this slave")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    server = SlaveServer(host=args.host, port=args.port, name=args.name)

    def signal_handler(sig, frame):
        log.info("Interrupt received, shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
