"""
gallery_server.py — Mini HTTP server to expose validation images remotely.

Runs `python -m http.server` style server (in-process via http.server module)
on a chosen port, serving a directory of validation images. Optionally exposes
it via ngrok for remote access from phone/tablet.

This is the "method 2" alternative to patching NeoSR — zero risk to the
training process.
"""
import os
import sys
import threading
import socket
import subprocess
import time
import shutil
import http.server
import socketserver
from urllib.parse import quote


# Custom directory listing template — mobile-friendly with auto-refresh
HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>USR Studio — Galerie Validation</title>
<style>
body {{ margin:0; padding:10px; background:#1a1a2e; color:#eee;
       font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
h1 {{ color:#3498db; font-size:18px; margin:5px 0 10px; }}
.info {{ color:#888; font-size:12px; margin-bottom:15px; }}
.refresh-bar {{ position:sticky; top:0; background:#1a1a2e; padding:5px 0;
                z-index:10; border-bottom:1px solid #333; margin-bottom:10px; }}
.refresh-bar button {{ background:#3498db; color:white; border:none;
                       padding:8px 16px; border-radius:4px; font-size:14px;
                       cursor:pointer; margin-right:8px; }}
.refresh-bar button:hover {{ background:#2980b9; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
         gap:12px; }}
.card {{ background:#2B2B3B; border-radius:8px; overflow:hidden;
         border:1px solid #3a3a4f; }}
.card img {{ width:100%; display:block; cursor:zoom-in;
             image-rendering:pixelated; }}
.card .name {{ padding:8px 10px; font-size:11px; color:#ccc; word-break:break-all; }}
.card .meta {{ padding:0 10px 8px; font-size:10px; color:#888; }}
.zoom {{ position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.95);
         display:none; justify-content:center; align-items:center; z-index:100;
         cursor:zoom-out; padding:10px; }}
.zoom.active {{ display:flex; }}
.zoom img {{ max-width:100%; max-height:100%; image-rendering:pixelated; }}
</style>
</head><body>
<div class="refresh-bar">
  <button onclick="location.reload()">🔄 Rafraichir</button>
  <label><input type="checkbox" id="auto" onchange="toggleAuto()"> Auto-refresh 30s</label>
</div>
<h1>📁 {dir_name}</h1>
<div class="info">{count} fichier(s) — Tri par date (recents en haut)</div>
<div class="grid">
{cards}
</div>
<div class="zoom" id="zoom" onclick="this.classList.remove('active')">
  <img id="zoomImg">
</div>
<script>
function zoomImg(src) {{
  document.getElementById('zoomImg').src = src;
  document.getElementById('zoom').classList.add('active');
}}
let autoTimer = null;
function toggleAuto() {{
  if (document.getElementById('auto').checked) {{
    autoTimer = setInterval(() => location.reload(), 30000);
  }} else if (autoTimer) {{
    clearInterval(autoTimer); autoTimer = null;
  }}
}}
</script>
</body></html>
"""


class GalleryHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that renders an image gallery HTML for directories."""

    def list_directory(self, path):
        """Override to render a custom mobile-friendly gallery."""
        try:
            entries = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission")
            return None

        # Filter and sort by mtime descending
        img_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
        items = []
        for name in entries:
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                items.append((name, "dir", 0, 0))
            elif name.lower().endswith(img_exts):
                try:
                    mtime = os.path.getmtime(full)
                    size = os.path.getsize(full)
                    items.append((name, "img", mtime, size))
                except OSError:
                    pass

        # Sort: dirs first, then images by mtime desc
        items.sort(key=lambda x: (0 if x[1] == "dir" else 1, -x[2], x[0]))

        cards = []
        for name, kind, mtime, size in items:
            link = quote(name)
            if kind == "dir":
                cards.append(
                    f'<div class="card"><a href="{link}/" style="color:#3498db;'
                    f'display:block;padding:20px;text-align:center;text-decoration:none">'
                    f'📁 {name}</a></div>'
                )
            else:
                size_kb = size / 1024
                size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                date_str = time.strftime("%H:%M:%S", time.localtime(mtime))
                cards.append(
                    f'<div class="card">'
                    f'<img src="{link}" loading="lazy" onclick="zoomImg(this.src)">'
                    f'<div class="name">{name}</div>'
                    f'<div class="meta">{date_str} • {size_str}</div>'
                    f'</div>'
                )

        dir_name = os.path.basename(path.rstrip(os.sep)) or "/"
        body = HTML_TEMPLATE.format(
            dir_name=dir_name,
            count=len([i for i in items if i[1] == "img"]),
            cards="\n".join(cards) if cards else "<p style='color:#888'>Aucune image dans ce dossier.</p>"
        ).encode("utf-8")

        from io import BytesIO
        f = BytesIO()
        f.write(body)
        length = f.tell()
        f.seek(0)

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def log_message(self, format, *args):
        # Suppress default stdout logging
        pass


class GalleryServer:
    """Wraps the HTTP server in a daemon thread + optional ngrok tunnel."""

    def __init__(self):
        self.httpd = None
        self.thread = None
        self.serve_dir = None
        self.port = 0
        self.ngrok_proc = None
        self.ngrok_url = ""

    def find_free_port(self, start: int = 8765) -> int:
        """Find an available port starting from `start`."""
        for p in range(start, start + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", p))
                    return p
            except OSError:
                continue
        return start

    def start(self, serve_dir: str, port: int = 0, with_ngrok: bool = False) -> dict:
        """
        Start the server.

        Returns dict with: ok, port, local_url, ngrok_url, error
        """
        if self.httpd is not None:
            return {"ok": False, "error": "Serveur deja actif. Arretez-le d'abord."}
        if not os.path.isdir(serve_dir):
            return {"ok": False, "error": f"Dossier introuvable: {serve_dir}"}

        if port == 0:
            port = self.find_free_port()

        self.serve_dir = serve_dir
        self.port = port

        # Need to chdir for SimpleHTTPRequestHandler
        original_cwd = os.getcwd()
        try:
            os.chdir(serve_dir)
            self.httpd = socketserver.ThreadingTCPServer(
                ("", port), GalleryHandler
            )
            self.httpd.allow_reuse_address = True
        except OSError as e:
            os.chdir(original_cwd)
            return {"ok": False, "error": f"Port {port} occupe: {e}"}
        except Exception as e:
            os.chdir(original_cwd)
            return {"ok": False, "error": f"Erreur: {e}"}

        # Don't restore cwd — server needs it. Save for later restoration.
        self._original_cwd = original_cwd

        # Run server in background thread
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

        local_url = f"http://localhost:{port}"
        result = {"ok": True, "port": port, "local_url": local_url, "ngrok_url": ""}

        # Optional ngrok tunnel
        if with_ngrok:
            ng_url = self._start_ngrok(port)
            if ng_url:
                self.ngrok_url = ng_url
                result["ngrok_url"] = ng_url
            else:
                result["ngrok_warning"] = "Ngrok non trouve ou echec de lancement."

        return result

    def _start_ngrok(self, port: int) -> str:
        """Start an ngrok HTTP tunnel pointing to the given port. Returns the public URL or ''."""
        ngrok_cmd = shutil.which("ngrok") or shutil.which("ngrok.exe")
        if not ngrok_cmd:
            return ""
        try:
            # Launch ngrok in background
            self.ngrok_proc = subprocess.Popen(
                [ngrok_cmd, "http", str(port), "--log", "stdout", "--log-format", "json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            # Poll the local API to get the public URL
            time.sleep(2.5)
            import urllib.request
            import json
            for _ in range(8):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        for t in data.get("tunnels", []):
                            url = t.get("public_url", "")
                            if url.startswith("https://"):
                                return url
                except Exception:
                    time.sleep(1)
            return ""
        except Exception as e:
            print(f"[Ngrok] Erreur: {e}")
            return ""

    def stop(self):
        """Stop the server and ngrok if running. Resets all state."""
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None
            self.thread = None

        if self.ngrok_proc is not None:
            try:
                self.ngrok_proc.terminate()
                self.ngrok_proc.wait(timeout=3)
            except Exception:
                try:
                    self.ngrok_proc.kill()
                except Exception:
                    pass
            self.ngrok_proc = None

        # Reset all state so a fresh start() is clean
        self.ngrok_url = ""
        self.port = 0
        self.serve_dir = None

        # Restore original cwd
        if hasattr(self, "_original_cwd"):
            try:
                os.chdir(self._original_cwd)
            except Exception:
                pass
            try:
                del self._original_cwd
            except Exception:
                pass

    def is_running(self) -> bool:
        return self.httpd is not None

    def status(self) -> dict:
        running = self.is_running()
        return {
            "running": running,
            "port": self.port if running else 0,
            "serve_dir": self.serve_dir if running else "",
            "local_url": f"http://localhost:{self.port}" if (running and self.port) else "",
            "ngrok_url": self.ngrok_url if running else "",
        }


# Module-level singleton (only one server at a time per app)
_global_server = GalleryServer()


def get_server() -> GalleryServer:
    return _global_server
