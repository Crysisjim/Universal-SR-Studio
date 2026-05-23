"""
distributed_client.py — Master-side client for SR Slave communication
=====================================================================
Import into the main GUI app to discover and control slave workers.

Usage:
    from src.core.distributed_client import SlaveClient, discover_slaves

    # Discover slaves on LAN
    slaves = discover_slaves(port=5001, timeout=2.0)

    # Send commands
    client = SlaveClient("192.168.1.100", 5001)
    info = client.ping()
    gpu = client.gpu_info()
    client.start_training(config_content="...", engine="neosr", ...)
"""

import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class SlaveClient:
    """TCP client to communicate with a single SR Slave."""

    def __init__(self, host: str, port: int = 5001, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_command(self, cmd: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a JSON command and receive a JSON response."""
        request = {"cmd": cmd}
        if data:
            request["data"] = data

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            msg = (json.dumps(request) + "\n").encode("utf-8")
            sock.sendall(msg)

            # Receive response
            buf = ""
            while "\n" not in buf:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")

            sock.close()

            if buf.strip():
                return json.loads(buf.strip().split("\n")[0])
            return {"ok": False, "error": "Empty response"}

        except socket.timeout:
            return {"ok": False, "error": f"Timeout connecting to {self.host}:{self.port}"}
        except ConnectionRefusedError:
            return {"ok": False, "error": f"Connection refused: {self.host}:{self.port}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def ping(self) -> Dict[str, Any]:
        return self.send_command("ping")

    def gpu_info(self) -> Dict[str, Any]:
        return self.send_command("gpu_info")

    def benchmark(self, seconds: int = 10) -> Dict[str, Any]:
        return self.send_command("benchmark", {"seconds": seconds})

    def sync_dataset(self, source: str, name: str) -> Dict[str, Any]:
        return self.send_command("sync_dataset", {"source": source, "name": name})

    def start_training(self, config_content: str, config_name: str = "train.yml",
                       master_addr: str = "127.0.0.1", master_port: int = 29500,
                       node_rank: int = 0, nnodes: int = 1, nproc_per_node: int = 1,
                       engine: str = "neosr", python_path: str = "python") -> Dict[str, Any]:
        return self.send_command("start_training", {
            "config_content": config_content,
            "config_name": config_name,
            "master_addr": master_addr,
            "master_port": master_port,
            "node_rank": node_rank,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "engine": engine,
            "python_path": python_path,
        })

    def stop_training(self) -> Dict[str, Any]:
        return self.send_command("stop_training")

    def status(self) -> Dict[str, Any]:
        return self.send_command("status")

    def get_logs(self, count: int = 50) -> Dict[str, Any]:
        return self.send_command("get_logs", {"count": count})

    def shutdown(self) -> Dict[str, Any]:
        return self.send_command("shutdown")


# ─── LAN Discovery ──────────────────────────────────────────
def discover_slaves(port: int = 5001, timeout: float = 1.5,
                    subnet: str = "", max_workers: int = 50) -> List[Dict[str, Any]]:
    """
    Scan LAN for active SR Slaves by sending ping to each IP.
    Returns list of responding slaves with their info.
    """
    if not subnet:
        # Auto-detect subnet from local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            subnet = ".".join(local_ip.split(".")[:3])
        except Exception:
            subnet = "192.168.1"

    found: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _probe(ip: str):
        client = SlaveClient(ip, port, timeout=timeout)
        result = client.ping()
        if result.get("ok"):
            with lock:
                found.append({
                    "ip": ip, "port": port,
                    "name": result.get("name", ip),
                    "version": result.get("version", "?"),
                    "platform": result.get("platform", "?"),
                    "hostname": result.get("hostname", "?"),
                })

    threads = []
    for i in range(1, 255):
        ip = f"{subnet}.{i}"
        t = threading.Thread(target=_probe, args=(ip,), daemon=True)
        threads.append(t)
        t.start()

        # Limit concurrent threads
        if len(threads) >= max_workers:
            for tt in threads:
                tt.join(timeout=timeout + 0.5)
            threads.clear()

    for t in threads:
        t.join(timeout=timeout + 0.5)

    return sorted(found, key=lambda x: x["ip"])


# ─── Distributed Training Orchestrator ───────────────────────
class DistributedOrchestrator:
    """
    High-level orchestrator for multi-node training.
    Handles: discovery → benchmark → dataset sync → launch → monitor.
    """

    def __init__(self, master_ip: str, master_port: int = 29500,
                 slave_port: int = 5001):
        self.master_ip = master_ip
        self.master_port = master_port
        self.slave_port = slave_port
        self.slaves: List[Dict[str, Any]] = []
        self.clients: Dict[str, SlaveClient] = {}

    def discover(self, timeout: float = 2.0) -> List[Dict[str, Any]]:
        """Discover and register all slaves on LAN."""
        self.slaves = discover_slaves(port=self.slave_port, timeout=timeout)
        self.clients = {s["ip"]: SlaveClient(s["ip"], s["port"]) for s in self.slaves}
        return self.slaves

    def benchmark_all(self, seconds: int = 5) -> Dict[str, Any]:
        """Benchmark all slaves and compute relative performance."""
        results = {}
        for ip, client in self.clients.items():
            result = client.benchmark(seconds)
            if result.get("ok") and "benchmarks" in result:
                total_ops = sum(b["ops_per_second"] for b in result["benchmarks"])
                results[ip] = {"benchmarks": result["benchmarks"], "total_ops": total_ops}
            else:
                results[ip] = {"error": result.get("error", "Unknown")}

        # Compute relative weights for batch size distribution
        valid = {k: v for k, v in results.items() if "total_ops" in v}
        if valid:
            max_ops = max(v["total_ops"] for v in valid.values())
            for ip in valid:
                valid[ip]["weight"] = round(valid[ip]["total_ops"] / max_ops, 3)

        return results

    def sync_all(self, source: str, name: str = "dataset",
                 callback=None) -> Dict[str, Any]:
        """Sync dataset to all slaves."""
        results = {}
        for ip, client in self.clients.items():
            if callback:
                callback(f"Syncing to {ip}...")
            result = client.sync_dataset(source, name)
            results[ip] = result
        return results

    def start_all(self, config_content: str, config_name: str = "train.yml",
                  engine: str = "neosr", nproc_per_node: int = 1,
                  python_path: str = "python") -> Dict[str, Any]:
        """Launch distributed training on all slaves."""
        nnodes = len(self.clients) + 1  # +1 for master
        results = {}

        for i, (ip, client) in enumerate(self.clients.items()):
            node_rank = i + 1  # Master is rank 0
            result = client.start_training(
                config_content=config_content,
                config_name=config_name,
                master_addr=self.master_ip,
                master_port=self.master_port,
                node_rank=node_rank,
                nnodes=nnodes,
                nproc_per_node=nproc_per_node,
                engine=engine,
                python_path=python_path,
            )
            results[ip] = result

        return results

    def stop_all(self) -> Dict[str, Any]:
        """Stop training on all slaves."""
        results = {}
        for ip, client in self.clients.items():
            results[ip] = client.stop_training()
        return results

    def status_all(self) -> Dict[str, Any]:
        """Get training status from all slaves."""
        results = {}
        for ip, client in self.clients.items():
            results[ip] = client.status()
        return results


# ─── Example Workflow ────────────────────────────────────────
if __name__ == "__main__":
    print("=== SR Distributed Client — Test ===\n")

    # 1. Discover slaves
    print("Scanning LAN for slaves...")
    slaves = discover_slaves(timeout=1.5)
    print(f"Found {len(slaves)} slave(s):")
    for s in slaves:
        print(f"  {s['ip']} — {s['name']} ({s['platform']})")

    if not slaves:
        print("No slaves found. Make sure neo_sr_slave.py is running on worker PCs.")
        exit(0)

    # 2. Get GPU info from first slave
    client = SlaveClient(slaves[0]["ip"], slaves[0]["port"])
    gpu = client.gpu_info()
    if gpu.get("ok"):
        for g in gpu.get("gpus", []):
            print(f"  GPU: {g['name']} — {g['total_vram_mb']} MB VRAM")

    # 3. Benchmark
    print("\nBenchmarking...")
    bench = client.benchmark(5)
    print(f"  Result: {bench}")

    print("\nDone!")
