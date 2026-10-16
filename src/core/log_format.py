"""
log_format.py — UI-agnostic training-log formatter (extracted from the CustomTkinter
tab_run so the Qt UI renders the *exact same* uniform, detailed, colored log
regardless of engine / arch / losses).

Takes raw traiNNer-redux / NeoSR (rich-logging) stdout and emits clean NeoSR-style
lines carrying ANSI color codes. The UI converts the ANSI codes to its own coloring
(QTextCharFormat for Qt, tk tags for CTk). The formatter is stateful (multi-line
records are buffered and flushed as one tidy line) and tracks best PSNR/SSIM so the
UI can update its cards.

Usage:
    fmt = ReduxLogFormatter()
    fmt.current_iter = 80000            # UI keeps this in sync from parsed iter
    fmt.set_val_dirs(lq_dir, gt_dir)    # optional, enriches the validation box
    ansi_text = fmt.feed(raw_chunk)     # render; then read fmt.best_psnr / best_ssim
"""
from __future__ import annotations

import os
import re


def _t(fr: str, en: str) -> str:
    """French or English string following the Qt UI language (French if unknown)."""
    try:
        from qt_ui.i18n import get_lang
        return en if get_lang() == "en" else fr
    except Exception:
        return fr


class ReduxLogFormatter:
    # Verbose Redux init blocks to suppress entirely (big YAML dumps, system info repeats)
    _REDUX_SUPPRESS = (
        "Diff with default config",
        "Training statistics for",
    )
    # Verbose init records to suppress (loss/optimizer/scheduler creation details)
    _REDUX_SUPPRESS_PREFIX = (
        "Loss ",
        "Scheduler ",
    )

    def __init__(self):
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_psnr_iter = 0
        self.best_ssim_iter = 0
        self.current_iter = 0
        # Last parsed iter-record metrics (UI reads these to fill the dashboard cards
        # + the metrics graph — robust because we parse the *joined* record, not the
        # raw multi-line fragments that a naive regex on the stream would miss).
        self.metrics = {}
        # Every validation-box reading, as (iter, value) — the UI drains these into
        # the Metrics dialog's PSNR/SSIM graphs (best_psnr/best_ssim above only track
        # the single best value for the dashboard card, not the full history).
        self.psnr_points = []
        self.ssim_points = []
        # USRS texture metrics readings (usrs_tex_mae...), drained by the UI like psnr_points
        self.extra_points = {}
        self._val_lq_dataset_dir = ""
        self._val_dataset_dir = ""
        self._reset_redux_buf()

    # ── public API ──────────────────────────────────────────────────────────────
    def set_val_dirs(self, lq_dir: str = "", gt_dir: str = ""):
        self._val_lq_dataset_dir = lq_dir or ""
        self._val_dataset_dir = gt_dir or ""

    def feed(self, text: str) -> str:
        """Normalize a raw chunk → clean ANSI-colored text (engine-agnostic)."""
        cleaned_chunks = []
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                cleaned_chunks.append(raw_line)
                continue
            for nline in self._normalize_redux_line(line):
                if nline:
                    cleaned_chunks.append(nline + "\n")
        return "".join(cleaned_chunks)

    def flush(self) -> str:
        """Flush any pending buffered record (call before printing a non-engine status
        line so the last iter / validation box isn't swallowed)."""
        out = []
        if self._redux_buf_active and self._redux_buf:
            f = self._flush_redux_buf()
            if f:
                out.append(f)
        if getattr(self, "_neosr_val_active", False):
            b = self._flush_neosr_val(self._neosr_val_time)
            self._neosr_val_active = False
            if b:
                out.append(b)
        self._redux_draining = False
        return ("\n".join(out) + "\n") if out else ""

    def reset(self):
        # Wipe cross-run state too — otherwise a fresh training that legitimately
        # starts low never displays as "best" until it beats the PREVIOUS run's
        # high-water mark (or never, if the old one already converged), leaving
        # the UI stuck on stale numbers/iter from before this run started.
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_psnr_iter = 0
        self.best_ssim_iter = 0
        self.current_iter = 0
        self.metrics = {}
        self.psnr_points = []
        self.ssim_points = []
        self.extra_points = {}
        self._reset_redux_buf()

    # ── state ───────────────────────────────────────────────────────────────────
    def _reset_redux_buf(self):
        self._redux_buf = ""
        self._redux_buf_active = False
        self._redux_buf_kind = ""
        self._redux_buf_lines = 0
        self._redux_buf_time = ""
        self._redux_buf_level = ""
        self._redux_draining = False
        self._val_images_seen = []
        self._val_total = 0
        self._val_in_progress = False
        self._neosr_val_active = False
        self._neosr_val_name = ""
        self._neosr_val_time = ""
        self._neosr_val_psnr = None
        self._neosr_val_ssim = None
        self._neosr_val_img_count = 0

    # ── line normalization ──────────────────────────────────────────────────────
    def _normalize_redux_line(self, line: str):
        """Convert noisy Redux/rich-logging output into clean NeoSR-style lines."""
        stripped_line = line.strip()
        if stripped_line.startswith("forrtl:") or re.match(r'^(KERNELBASE|KERNEL32|ntdll)\.dll', stripped_line):
            return []

        if re.match(r'^Test\s+\S+:\s+\d+%', line):
            m_t = re.match(r'^Test\s+(\S+):\s+\d+%.*?(\d+)/(\d+)', line)
            if m_t:
                img_name = m_t.group(1)
                self._val_total = int(m_t.group(3))
                if img_name not in self._val_images_seen:
                    self._val_images_seen.append(img_name)
            return []
        if re.search(r'\d+/\d+\s+\[.*?image', line, re.IGNORECASE):
            m_total = re.search(r'\d+/(\d+)', line)
            if m_total:
                self._val_total = max(getattr(self, '_val_total', 0), int(m_total.group(1)))
            return []

        # --- NeoSR timestamp format: "DD-MM-YYYY HH:MM AM|PM | INFO: content" ---
        neosr_m = re.match(r"^(\d{2}-\d{2}-\d{4})\s+(\d{1,2}:\d{2})\s+(AM|PM)\s*\|\s*INFO:\s*(.*)$", line, re.IGNORECASE)
        if neosr_m:
            time_str = f"{neosr_m.group(2)} {neosr_m.group(3).upper()}"
            content = re.sub(r'[\x1b\033]\[[0-9;]*m', '', neosr_m.group(4)).strip()
            output = []
            C = "\x1b[36m"; G = "\x1b[32m"; Y = "\x1b[33m"; X = "\x1b[0m"
            if self._neosr_val_active:
                box = self._flush_neosr_val(self._neosr_val_time)
                if box:
                    output.append(box)
                self._neosr_val_active = False
            if not content:
                return output
            if re.search(r"\[\s*epoch:\s*\d+", content):
                output.append(self._format_redux_record(time_str, content))
                return output
            m_nval = re.match(r"Validation\s+(.+)", content)
            if m_nval:
                self._neosr_val_active = True
                self._neosr_val_time = time_str
                self._neosr_val_name = m_nval.group(1).strip()
                self._neosr_val_psnr = None
                self._neosr_val_ssim = None
                return output
            m_nval_count = re.search(r"Number of val images/folders:\s*(\d+)", content)
            if m_nval_count:
                self._neosr_val_img_count = int(m_nval_count.group(1))

            if "Saving models and training states" in content:
                output.append(f"{C}[{time_str}]{X} {G}INFO: {content}{X}")
                return output
            if "Resuming training from epoch" in content or "Start training from epoch" in content:
                output.append(f"{C}[{time_str}]{X} {G}INFO: {content}{X}")
                return output
            if "Interrupted" in content or "interrupted" in content:
                output.append(f"{C}[{time_str}]{X} {Y}INFO: {content}{X}")
                return output
            output.append(f"{C}[{time_str}]{X} {C}INFO:{X} {content}")
            return output

        if getattr(self, '_neosr_val_active', False):
            stripped = re.sub(r'[\x1b\033]\[[0-9;]*m', '', line)
            m_psnr = re.search(r'psnr:\s*([0-9.]+)', stripped, re.IGNORECASE)
            m_ssim = re.search(r'ssim:\s*([0-9.]+)', stripped, re.IGNORECASE)
            if m_psnr:
                self._neosr_val_psnr = m_psnr.group(1)
            if m_ssim:
                self._neosr_val_ssim = m_ssim.group(1)
            if self._neosr_val_psnr is not None and self._neosr_val_ssim is not None:
                box = self._flush_neosr_val(self._neosr_val_time)
                self._neosr_val_active = False
                return [box] if box else []
            return []

        prefix_re = re.compile(r"^\[(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(.*)$")
        src_re = re.compile(r"\s+\w+\.py:\d+\s*$")

        m = prefix_re.match(line)
        if m:
            output_lines = []
            self._redux_draining = False
            if self._redux_buf_active and self._redux_buf:
                flushed = self._flush_redux_buf()
                if flushed:
                    output_lines.append(flushed)

            time_str = m.group(2)
            level = m.group(3)
            content = src_re.sub("", m.group(4)).rstrip()

            if re.match(r'^(?:(?:images?|it|s)\/s\s*)+$', content.strip(), re.IGNORECASE):
                return output_lines

            is_iter = bool(re.search(r"\[\s*epoch:\s*\d+", content)) or "[ iter:" in content or "[iter:" in content
            is_suppress = (
                any(content.startswith(p) for p in self._REDUX_SUPPRESS)
                or any(content.startswith(p) for p in self._REDUX_SUPPRESS_PREFIX)
            )

            if not is_iter and not is_suppress and re.match(
                    r'Validation\s+.+?(?:#\s*)?psnr\s*[:=]\s*[0-9.]+', content, re.IGNORECASE):
                self._redux_buf_time = time_str
                self._redux_buf_level = level
                self._redux_buf = content
                self._redux_buf_lines = 1
                self._redux_buf_kind = "info"
                box = self._flush_redux_buf()
                if box:
                    output_lines.append(box)
                return output_lines

            self._redux_buf_time = time_str
            self._redux_buf_level = level
            self._redux_buf = content
            self._redux_buf_active = True
            self._redux_buf_lines = 1
            self._redux_buf_kind = "iter" if is_iter else ("suppress" if is_suppress else "info")
            return output_lines

        if self._redux_buf_active:
            if self._redux_buf_kind == "suppress":
                return []
            src_re = re.compile(r"\s+\w+\.py:\d+\s*$")
            stripped = src_re.sub("", line.strip()).rstrip()
            if not stripped:
                return []
            self._redux_buf_lines += 1
            self._redux_buf += " " + stripped
            if self._redux_buf_kind == "iter":
                has_scale_d = bool(re.search(r"scale_d:\s*[0-9.eE+\-]+", self._redux_buf))
                if has_scale_d or self._redux_buf_lines > 14:
                    flushed = self._flush_redux_buf()
                    self._redux_draining = True
                    return [flushed] if flushed else []
                return []
            else:
                if self._redux_buf_lines > 25:
                    saved_time = self._redux_buf_time
                    saved_level = self._redux_buf_level
                    flushed = self._flush_redux_buf()
                    self._redux_buf_active = True
                    self._redux_buf_kind = "info"
                    self._redux_buf_time = saved_time
                    self._redux_buf_level = saved_level
                    return [flushed] if flushed else []
                return []

        if getattr(self, '_redux_draining', False):
            if not re.match(r"^\[[\d/]+\s+[\d:]+\]", line):
                return []

        return [line]

    def _flush_redux_buf(self) -> str:
        C = "\x1b[36m"; G = "\x1b[32m"; Y = "\x1b[33m"; X = "\x1b[0m"
        kind = self._redux_buf_kind
        time_str = self._redux_buf_time
        level = self._redux_buf_level
        raw = self._redux_buf

        self._redux_buf = ""
        self._redux_buf_active = False
        self._redux_buf_kind = ""
        self._redux_buf_lines = 0

        if kind == "suppress":
            return ""
        if kind == "iter":
            return self._format_redux_record(time_str, raw)

        content = re.sub(r"\s+", " ", raw).strip()
        if re.match(r'^(?:(?:images?|it|s)\/s\s*)+$', content, re.IGNORECASE):
            return ""
        content = re.sub(r'[A-Za-z]:[\\\/](?:[^\\\/\n,]+[\\\/]){2,}([^\\\/\n,]+)', r'...\\\1', content)

        m_saving = re.search(r'Saving\s+(\d+)\s+validation\s+images', content, re.IGNORECASE)
        if m_saving:
            self._val_in_progress = True
            self._val_total = int(m_saving.group(1))
            return (f"{C}[{time_str}]{X} 🔍 "
                    + _t(f"Traitement de {self._val_total} images de validation...",
                         f"Processing {self._val_total} validation images..."))

        if re.match(r'Test\s+\S+:\s+\d+%', content):
            m_t = re.match(r'Test\s+(\S+):\s+\d+%.*?(\d+)/(\d+)', content)
            if m_t:
                self._val_total = int(m_t.group(3))
            return ""
        if getattr(self, '_val_in_progress', False) and re.search(r'\d+/\d+\s+\[.*?image', content, re.IGNORECASE):
            m_total = re.search(r'\d+/(\d+)', content)
            if m_total:
                self._val_total = max(getattr(self, '_val_total', 0), int(m_total.group(1)))
            return ""

        m_val = re.match(r'Validation\s+(.+?)\s*(?:#\s*)?psnr\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE)
        if m_val:
            val_name = m_val.group(1).strip().rstrip(":").strip()
            psnr_v = m_val.group(2)
            ssim_m = (re.search(r'#\s*ssim\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE) or
                      re.search(r'\bssim\s*[:=]\s*([0-9.]+)', content, re.IGNORECASE))
            ssim_v = ssim_m.group(1) if ssim_m else None
            extra = {}
            try:
                from src.core.val_extra_metrics import KEYS as _EXTRA_KEYS
            except Exception:
                _EXTRA_KEYS = ()
            for _k in _EXTRA_KEYS:
                _m = re.search(r'#\s*' + _k + r'\s*[:=]\s*(-?[0-9.]+)', content, re.IGNORECASE)
                if _m:
                    try:
                        extra[_k] = float(_m.group(1))
                        self.extra_points.setdefault(_k, []).append((self.current_iter, extra[_k]))
                    except Exception:
                        pass

            try:
                pf = float(psnr_v)
                if 0 < pf < 100:
                    self.psnr_points.append((self.current_iter, pf))
                    if pf > self.best_psnr:
                        self.best_psnr = pf
                        self.best_psnr_iter = self.current_iter
            except Exception:
                pass
            if ssim_v:
                try:
                    sf = float(ssim_v)
                    if 0 < sf <= 1.0:
                        self.ssim_points.append((self.current_iter, sf))
                        if sf > self.best_ssim:
                            self.best_ssim = sf
                            self.best_ssim_iter = self.current_iter
                except Exception:
                    pass

            images = list(getattr(self, '_val_images_seen', []))
            real_total = max(getattr(self, '_val_total', 0), getattr(self, '_neosr_val_img_count', 0))
            total = real_total or len(images)
            _ds_dir = (getattr(self, '_val_lq_dataset_dir', '') or getattr(self, '_val_dataset_dir', ''))
            if _ds_dir and os.path.isdir(_ds_dir):
                _img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
                try:
                    _all_stems = sorted(
                        (os.path.splitext(fn)[0] for fn in os.listdir(_ds_dir)
                         if os.path.splitext(fn)[1].lower() in _img_exts),
                        key=str.lower)
                    if len(_all_stems) > total:
                        total = len(_all_stems)
                    _seen_set = set(images)
                    for _stem in _all_stems:
                        if _stem not in _seen_set:
                            images.append(_stem)
                            _seen_set.add(_stem)
                except Exception:
                    pass
            box = self._render_validation_box(val_name, psnr_v, ssim_v, images, total, time_str, extra)
            self._val_images_seen = []
            self._val_total = 0
            self._val_in_progress = False
            return box

        if "\U0001f680" in content and "traiNNer-redux" in content:
            gpu = re.search(r"Name:\s*(NVIDIA\s+GeForce\s+[^\d]*\d+[^\s]*(?:\s+Ti)?)", content)
            vram = re.search(r"Total VRAM:\s*([0-9.]+\s*GB)", content)
            pytorch = re.search(r"PyTorch:\s*([0-9.+a-z]+)", content)
            parts = ["\U0001f680 traiNNer-redux: good luck! \U0001f680"]
            if gpu:   parts.append(f"GPU: {gpu.group(1).strip()}")
            if vram:  parts.append(f"VRAM: {vram.group(1)}")
            if pytorch: parts.append(f"PyTorch: {pytorch.group(1)}")
            content = " | ".join(parts)

        m_ds = re.match(r"(Dataset \w+ - \w+ Dataset is built\.)\s*Number of train images:\s*([\d,]+)", content)
        if m_ds:
            content = f"{m_ds.group(1)} ({m_ds.group(2)} images)"

        if "Resuming training from epoch" in content:
            return f"{C}[{time_str}]{X} {G}{level}: {content}{X}"
        if "Saving models and training states" in content:
            return f"{C}[{time_str}]{X} {G}{level}: {content}{X}"
        if "User interrupted" in content or "Preparing to save state" in content:
            return f"{C}[{time_str}]{X} {Y}{level}: {content}{X}"

        return f"{C}[{time_str}]{X} {C}{level}:{X} {content}"

    def _render_validation_box(self, val_name, psnr_v, ssim_v, images, total, time_str, extra=None) -> str:
        # PSNR = teal (\x1b[96m → #1abc9c), SSIM = blue (\x1b[34m → #3498db) — matches the
        # dashboard cards so the colors correspond between the log box and the cards.
        C = "\x1b[36m"; M = "\x1b[35m"; G = "\x1b[32m"; Y = "\x1b[96m"; B = "\x1b[34m"; X = "\x1b[0m"
        FR = "\x1b[35m"   # mauve frame (border lines only)
        PK = "\x1b[95m"   # pink (the images-processed count)
        MAX_ROWS = 7
        n_imgs = len(images)
        n_cols = max(1, (n_imgs + MAX_ROWS - 1) // MAX_ROWS) if n_imgs else 1
        max_nm_len = min(max((len(nm) for nm in images), default=10), 24)
        cell_w = max_nm_len + 7
        header_vis = f"  VALIDATION — {val_name}  [{time_str}]"
        W = max(62, n_cols * cell_w, len(header_vis) + 2)

        def hline(left="╠", right="╣"):
            return f"{FR}{left}{'═' * W}{right}{X}"

        def row(vis: str, col: str = None) -> str:
            if col is None:
                col = vis
            pad = max(0, W - len(vis))
            return f"{FR}║{X}{col}{' ' * pad}{FR}║{X}"

        header_col = f"  {C}VALIDATION{X} — {C}{val_name}{X}  [{time_str}]"
        lines = [hline("╔", "╗"), row(header_vis, header_col), hline()]

        _cols = [images[i * MAX_ROWS:(i + 1) * MAX_ROWS] for i in range(n_cols)]
        n_rows = max((len(c) for c in _cols), default=0)
        for r_idx in range(n_rows):
            row_vis = ""
            row_col = ""
            for c_idx, col_items in enumerate(_cols):
                if r_idx < len(col_items):
                    nm = col_items[r_idx]
                    if len(nm) > max_nm_len:
                        nm = nm[:max_nm_len - 1] + "…"
                    row_vis += f"  ✓  {nm:<{max_nm_len}}  "
                    row_col += f"  {G}✓{X}  {nm:<{max_nm_len}}  "  # names in default (white) text
                else:
                    row_vis += " " * cell_w
                    row_col += " " * cell_w
            lines.append(row(row_vis, row_col))

        lines.append(hline())
        psnr_vis = f"  Best PSNR : {psnr_v} dB"
        psnr_col = f"  Best PSNR : {Y}{psnr_v} dB{X}"
        lines.append(row(psnr_vis, psnr_col))
        if ssim_v:
            ssim_vis = f"  Best SSIM : {ssim_v}"
            ssim_col = f"  Best SSIM : {B}{ssim_v}{X}"
            lines.append(row(ssim_vis, ssim_col))
        if extra:
            try:
                from src.core import val_extra_metrics as _vem
                for _k, _v in extra.items():
                    _lbl = _vem.short_label(_k)
                    _fmtv = f"{_v:.3f}" if _k.startswith("usrs_hf") else f"{_v:.2f}"
                    _vis = f"  {_lbl:<15}: {_fmtv}"
                    lines.append(row(_vis, f"  {_lbl:<15}: {Y}{_fmtv}{X}"))
            except Exception:
                pass
        if total > 0 or n_imgs > 0:
            shown = n_imgs if n_imgs else total
            cnt_total = total if total > 0 else n_imgs
            _lab = _t("Images traitées", "Images done")
            count_vis = f"  {_lab} : {shown} / {cnt_total}"
            count_col = f"  {_lab} : {PK}{shown} / {cnt_total}{X}"  # pink count
            lines.append(row(count_vis, count_col))
        lines.append(hline("╚", "╝"))
        return "\n".join(lines)

    def _flush_neosr_val(self, time_str: str) -> str:
        name = getattr(self, '_neosr_val_name', 'ValSet')
        psnr_v = getattr(self, '_neosr_val_psnr', None)
        ssim_v = getattr(self, '_neosr_val_ssim', None)
        if psnr_v is None:
            return ""
        try:
            pf = float(psnr_v)
            if pf > 0 and pf > self.best_psnr:
                self.best_psnr = pf
                self.best_psnr_iter = self.current_iter
        except Exception:
            pass
        if ssim_v:
            try:
                sf = float(ssim_v)
                if sf > 0 and sf > self.best_ssim:
                    self.best_ssim = sf
                    self.best_ssim_iter = self.current_iter
            except Exception:
                pass
        img_count = getattr(self, '_neosr_val_img_count', 0)
        images = []
        _ds_dir = (getattr(self, '_val_lq_dataset_dir', '') or getattr(self, '_val_dataset_dir', ''))
        if _ds_dir and os.path.isdir(_ds_dir):
            _img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
            try:
                images = sorted(
                    (os.path.splitext(fn)[0] for fn in os.listdir(_ds_dir)
                     if os.path.splitext(fn)[1].lower() in _img_exts),
                    key=str.lower)
            except Exception:
                pass
        total = max(img_count, len(images))
        return self._render_validation_box(name, psnr_v, ssim_v, images, total, time_str)

    def _format_redux_record(self, time_str: str, raw: str) -> str:
        C  = "\x1b[36m"; R = "\x1b[31m"; P = "\x1b[35m"; G = "\x1b[32m"; Y = "\x1b[33m"
        O  = "\x1b[38;5;208m"; BR = "\x1b[38;5;130m"; PK = "\x1b[95m"; X = "\x1b[0m"

        ep = re.search(r"epoch:\s*(\d+)", raw)
        it = re.search(r"iter:\s*([\d,]+)", raw)
        lr = re.search(r"lr:\s*\(?\s*([0-9.eE+\-]+)\s*\)?", raw)
        perf = re.search(r"(?:perf|performance):\s*([0-9.]+)\s*it/s", raw)
        eta = re.search(r"eta:\s*([^\]]+?)\s*\]", raw)
        vram = re.search(r"VRAM:\s*([0-9.]+)\s*GB", raw)
        lg_total = re.search(r"l_g_total:\s*([0-9.eE+\-]+)", raw)
        lg_components = re.findall(r"l_g_(\w+):\s*([0-9.eE+\-]+)", raw)
        ld_total = re.search(r"l_d_total:\s*([0-9.eE+\-]+)", raw)
        ld_real  = re.search(r"(?:l_)?d_real:\s*([0-9.eE+\-]+)", raw)
        ld_fake  = re.search(r"(?:l_)?d_fake:\s*([0-9.eE+\-]+)", raw)
        gn_g = re.search(r"grad_norm_g:\s*([0-9.eE+\-]+)", raw)
        gn_d = re.search(r"grad_norm_d:\s*([0-9.eE+\-]+)", raw)

        # Expose parsed values so the UI can fill its dashboard cards + graph reliably.
        m = {}
        if ep:   m["epoch"] = ep.group(1)
        if it:
            m["iter"] = int(it.group(1).replace(",", ""))
            self.current_iter = m["iter"]
        if perf: m["perf"] = perf.group(1)
        if lr:   m["lr"] = lr.group(1)
        if eta:  m["eta"] = eta.group(1).strip()
        if lg_total: m["loss_g"] = lg_total.group(1)
        if ld_total: m["loss_d"] = ld_total.group(1)
        elif ld_real: m["loss_d"] = ld_real.group(1)
        if vram: m["vram"] = vram.group(1)
        if m:
            self.metrics = m

        parts = [f"{C}[{time_str}]{X} {C}INFO:{X}"]
        if ep and it:
            iter_clean = it.group(1).rstrip(",")
            parts.append(f"[ epoch: {O}{ep.group(1):>3}{X} ] [ iter: {G}{iter_clean:>7}{X} ]")
        if perf:
            parts.append(f"[ perf: {Y}{perf.group(1)} it/s{X} ]")
        if lr:
            try:
                parts.append(f"[ lr: {BR}{float(lr.group(1)):.2e}{X} ]")
            except ValueError:
                parts.append(f"[ lr: {BR}{lr.group(1)}{X} ]")
        if eta:
            parts.append(f"[ eta: {eta.group(1).strip()} ]")

        _KEY_NAMES = {"total", "gan", "charbonnier", "ldl1", "ff", "focal", "real"}

        def _abbrev_g(name: str) -> str:
            _MAP = {"charbonnier": "charb", "ldl1": "ldl1", "ff": "ff",
                    "gan": "gan", "real": "real", "focal": "focal", "total": "total"}
            if name in _MAP:
                return _MAP[name]
            m = re.match(r"perceptual(?:_conv)?_?(\w+)", name)
            if m:
                return f"pc_{m.group(1)}"
            return name[:8] if len(name) > 8 else name

        loss_g_key = []
        loss_g_detail = []
        if lg_total:
            loss_g_key.append(f"total: {R}{lg_total.group(1)}{X}")
        for comp_name, comp_val in lg_components:
            if comp_name == "total":
                continue
            abbr = _abbrev_g(comp_name)
            piece = f"{abbr}: {R}{comp_val}{X}"
            if comp_name in _KEY_NAMES:
                loss_g_key.append(piece)
            else:
                loss_g_detail.append(piece)
        if loss_g_key:
            parts.append("[ " + " | ".join(loss_g_key) + " ]")

        line2_parts = []
        if loss_g_detail:
            line2_parts.append("[ " + " | ".join(loss_g_detail) + " ]")
        loss_d_pieces = []
        if ld_total:
            loss_d_pieces.append(f"d_total: {P}{ld_total.group(1)}{X}")
        else:
            if ld_real:
                loss_d_pieces.append(f"d_real: {P}{ld_real.group(1)}{X}")
            if ld_fake:
                loss_d_pieces.append(f"d_fake: {P}{ld_fake.group(1)}{X}")
        if loss_d_pieces:
            line2_parts.append("[ " + " | ".join(loss_d_pieces) + " ]")
        if vram:
            line2_parts.append(f"[ VRAM: {C}{vram.group(1)} GB{X} ]")
        gn_pieces = []
        if gn_g:
            gn_pieces.append(f"|g|: {PK}{gn_g.group(1)}{X}")
        if gn_d:
            gn_pieces.append(f"|d|: {PK}{gn_d.group(1)}{X}")
        if gn_pieces:
            line2_parts.append("[ " + " | ".join(gn_pieces) + " ]")

        if line2_parts:
            return " ".join(parts) + " " + " ".join(line2_parts)
        return " ".join(parts)
