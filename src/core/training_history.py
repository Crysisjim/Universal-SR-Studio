"""
training_history.py — SQLite database tracking all completed/interrupted trainings.

Records: name, engine, architecture, scale, dataset, total_iter, best_psnr, best_ssim,
duration, started_at, finished_at, status (running/completed/interrupted/failed), config_path,
avg_speed (it/s), gpu_name.
"""
import os
import sqlite3
import time
from datetime import datetime


DB_PATH = os.path.join(os.path.expanduser("~"), ".usr_studio_training_history.db")


def _get_conn():
    """Get connection to the history DB. Creates the table if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trainings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            engine TEXT,
            architecture TEXT,
            scale INTEGER,
            dataset_path TEXT,
            total_iter INTEGER,
            current_iter INTEGER DEFAULT 0,
            best_psnr REAL DEFAULT 0,
            best_ssim REAL DEFAULT 0,
            best_iter INTEGER DEFAULT 0,
            duration_seconds INTEGER DEFAULT 0,
            started_at INTEGER NOT NULL,
            finished_at INTEGER,
            status TEXT DEFAULT 'running',
            config_path TEXT,
            notes TEXT,
            batch_size INTEGER,
            patch_size INTEGER,
            lr TEXT,
            optimizer TEXT
        )
    """)
    conn.commit()
    # Safe schema migration: add new columns if they don't exist yet
    for col, definition in [
        ("avg_speed",    "REAL DEFAULT 0"),
        ("gpu_name",     "TEXT DEFAULT ''"),
        ("gan_mode",     "INTEGER DEFAULT 0"),   # 1 = GAN training
        ("losses",       "TEXT DEFAULT ''"),     # virgule-séparée, ex: "l_pix,l_perceptual,l_gan"
        ("upsampler",    "TEXT DEFAULT ''"),     # ex: "pixelshuffle", "nearest+conv"
        ("vram_peak_mb", "INTEGER DEFAULT 0"),   # pic VRAM en MB pendant l'entraînement
        ("power_avg_w",  "REAL DEFAULT 0"),      # conso élec moyenne GPU (W)
    ]:
        try:
            conn.execute(f"ALTER TABLE trainings ADD COLUMN {col} {definition}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
    return conn


def record_training_start(
    name: str, engine: str, architecture: str, scale: int,
    total_iter: int, config_path: str, dataset_path: str = "",
    batch_size: int = 0, patch_size: int = 0, lr: str = "", optimizer: str = "",
    gpu_name: str = "", gan_mode: int = 0, losses: str = "", upsampler: str = ""
) -> int:
    """Record the start of a training. Returns the row id."""
    conn = _get_conn()
    try:
        cursor = conn.execute("""
            INSERT INTO trainings
            (name, engine, architecture, scale, dataset_path, total_iter,
             started_at, status, config_path, batch_size, patch_size, lr, optimizer,
             gpu_name, gan_mode, losses, upsampler)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, engine, architecture, scale, dataset_path, total_iter,
              int(time.time()), config_path, batch_size, patch_size, lr, optimizer,
              gpu_name, gan_mode, losses, upsampler))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_training_progress(row_id: int, current_iter: int = None,
                              best_psnr: float = None, best_ssim: float = None,
                              best_iter: int = None, avg_speed: float = None):
    """Update progress for an ongoing training."""
    if row_id <= 0:
        return
    conn = _get_conn()
    try:
        updates = []
        params = []
        if current_iter is not None:
            updates.append("current_iter = ?")
            params.append(current_iter)
        if best_psnr is not None:
            updates.append("best_psnr = ?")
            params.append(best_psnr)
        if best_ssim is not None:
            updates.append("best_ssim = ?")
            params.append(best_ssim)
        if best_iter is not None:
            updates.append("best_iter = ?")
            params.append(best_iter)
        if avg_speed is not None:
            updates.append("avg_speed = ?")
            params.append(avg_speed)
        if not updates:
            return
        params.append(row_id)
        conn.execute(f"UPDATE trainings SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def record_training_end(row_id: int, status: str = "completed", notes: str = "",
                        avg_speed: float = None, vram_peak_mb: int = 0,
                        power_avg_w: float = 0.0):
    """Mark a training as finished. status: 'completed' / 'interrupted' / 'failed'."""
    if row_id <= 0:
        return
    conn = _get_conn()
    try:
        # Get started_at to compute duration
        cursor = conn.execute("SELECT started_at FROM trainings WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        duration = 0
        if row:
            duration = int(time.time()) - int(row[0])
        sets = ["finished_at = ?", "status = ?", "duration_seconds = ?", "notes = ?"]
        params = [int(time.time()), status, duration, notes]
        if avg_speed is not None and avg_speed > 0:
            sets.append("avg_speed = ?"); params.append(avg_speed)
        if vram_peak_mb > 0:
            sets.append("vram_peak_mb = ?"); params.append(int(vram_peak_mb))
        if power_avg_w > 0:
            sets.append("power_avg_w = ?"); params.append(float(power_avg_w))
        params.append(row_id)
        conn.execute(f"UPDATE trainings SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def get_recent_trainings(limit: int = 50) -> list:
    """Return list of recent trainings as dicts."""
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM trainings ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_trainings() -> list:
    """Return ALL training records as dicts (for export)."""
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM trainings ORDER BY started_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_training_by_id(row_id: int) -> dict:
    """Get a single training record."""
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM trainings WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_stats_by_architecture() -> list:
    """Aggregate stats grouped by architecture."""
    conn = _get_conn()
    try:
        cursor = conn.execute("""
            SELECT architecture,
                   COUNT(*) as count,
                   AVG(best_psnr) as avg_psnr,
                   MAX(best_psnr) as max_psnr,
                   AVG(duration_seconds) as avg_duration,
                   AVG(CASE WHEN avg_speed > 0 THEN avg_speed ELSE NULL END) as avg_speed
            FROM trainings
            WHERE status = 'completed' AND best_psnr > 0
            GROUP BY architecture
            ORDER BY avg_psnr DESC
        """)
        return [
            {
                "architecture": r[0] or "unknown",
                "count": r[1],
                "avg_psnr": r[2] or 0,
                "max_psnr": r[3] or 0,
                "avg_duration": r[4] or 0,
                "avg_speed": r[5] or 0,
            }
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()


def delete_training(row_id: int):
    """Remove a training record from history."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM trainings WHERE id = ?", (row_id,))
        conn.commit()
    finally:
        conn.close()


def delete_all_trainings():
    """Remove ALL training records from history."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM trainings")
        conn.commit()
    finally:
        conn.close()


def export_benchmark_txt() -> str:
    """Generate a formatted benchmark text report of all trainings."""
    trainings = get_all_trainings()
    lines = []
    lines.append("=" * 60)
    lines.append("BENCHMARK HISTORIQUE DES ENTRAINEMENTS — Universal SR Studio")
    lines.append(f"Exporte le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total enregistrements : {len(trainings)}")
    lines.append("=" * 60)
    lines.append("")

    for t in trainings:
        status = t.get("status", "?")
        psnr = t.get("best_psnr", 0)
        ssim = t.get("best_ssim", 0)
        speed = t.get("avg_speed", 0)
        gpu = t.get("gpu_name", "") or "?"
        gan = t.get("gan_mode", 0)
        losses_str = t.get("losses", "") or "—"
        upsampler = t.get("upsampler", "") or "—"
        vram_pk = t.get("vram_peak_mb", 0) or 0
        pwr = t.get("power_avg_w", 0.0) or 0.0
        mode_tag = " [GAN]" if gan else " [PSNR]"
        lines.append(f"[#{t['id']}] {t['name']}{mode_tag}  ({status.upper()})")
        lines.append(f"  Date         : {format_timestamp(t.get('started_at', 0))}")
        lines.append(f"  Engine       : {t.get('engine', '?')}")
        lines.append(f"  Architecture : {t.get('architecture', '?')}")
        lines.append(f"  Upsampler    : {upsampler}")
        lines.append(f"  Scale        : {t.get('scale', '?')}x")
        lines.append(f"  Iters        : {t.get('current_iter', 0)}/{t.get('total_iter', 0)}")
        lines.append(f"  Best PSNR    : {psnr:.4f} dB" if psnr else "  Best PSNR    : —")
        lines.append(f"  Best SSIM    : {ssim:.4f}" if ssim else "  Best SSIM    : —")
        lines.append(f"  Vitesse moy  : {speed:.3f} it/s" if speed else "  Vitesse moy  : —")
        lines.append(f"  Duree        : {format_duration(t.get('duration_seconds', 0))}")
        lines.append(f"  GPU          : {gpu}")
        lines.append(f"  VRAM pic     : {vram_pk // 1024:.1f} GB" if vram_pk else "  VRAM pic     : —")
        lines.append(f"  Conso GPU    : {pwr:.0f} W" if pwr else "  Conso GPU    : —")
        lines.append(f"  Batch/Patch  : {t.get('batch_size', '?')} / {t.get('patch_size', '?')}")
        lines.append(f"  LR           : {t.get('lr', '?')}")
        lines.append(f"  Optimizer    : {t.get('optimizer', '?')}")
        lines.append(f"  Losses       : {losses_str}")
        ds = t.get("dataset_path", "") or "?"
        lines.append(f"  Dataset      : {ds}")
        if t.get("notes"):
            lines.append(f"  Notes        : {t['notes']}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("FIN DU RAPPORT")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_duration(seconds: int) -> str:
    """Format seconds as 'Xj Xh Xmin'."""
    if not seconds:
        return "0s"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}min"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}min"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    return f"{days}j {hours}h"


def format_timestamp(ts: int) -> str:
    """Format unix timestamp as readable date."""
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
