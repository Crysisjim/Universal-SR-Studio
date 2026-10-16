"""
taskbar.py — Windows taskbar progress overlay (ITaskbarList3), pure ctypes COM VTable
calls, no pywin32 dependency. Direct port of the CTk app's taskbar controller — the
Qt port never had this at all, so a running training showed no progress on the
taskbar icon (useful when the window is minimized or behind other apps).

Usage:
    from qt_ui.taskbar import init_taskbar, set_taskbar_progress
    init_taskbar(self)                              # once, after the window is shown
    set_taskbar_progress(0.42, "normal")             # 0.0-1.0, state in
                                                      # normal/paused/error/indeterminate/none
"""
from __future__ import annotations

import sys
import ctypes


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


TBPF_NOPROGRESS = 0
TBPF_INDETERMINATE = 1
TBPF_NORMAL = 2
TBPF_ERROR = 4
TBPF_PAUSED = 8

_CLSID_TaskbarList = "{56FDF344-FD6D-11d0-958A-006097C9A090}"
_IID_ITaskbarList3 = "{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}"


class _TaskbarController:
    def __init__(self):
        self.ptr = None
        self.hwnd = 0
        if sys.platform != "win32":
            return
        try:
            ctypes.windll.ole32.CoInitialize(0)
            clsid = _GUID(); iid = _GUID()
            ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(_CLSID_TaskbarList), ctypes.byref(clsid))
            ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(_IID_ITaskbarList3), ctypes.byref(iid))
            self.ptr = ctypes.c_void_p()
            ctypes.windll.ole32.CoCreateInstance(
                ctypes.byref(clsid), 0, 1,  # CLSCTX_INPROC_SERVER
                ctypes.byref(iid), ctypes.byref(self.ptr))
        except Exception:
            self.ptr = None

    def set_hwnd(self, hwnd):
        self.hwnd = hwnd

    def _call_method(self, method_index, *args):
        if not self.ptr or not self.hwnd:
            return
        try:
            vtable = ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_void_p)).contents
            func_ptr = ctypes.cast(
                vtable.value + (method_index * ctypes.sizeof(ctypes.c_void_p)),
                ctypes.POINTER(ctypes.c_void_p)).contents
            arg_types = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint64 for _ in args]
            functype = ctypes.WINFUNCTYPE(ctypes.c_long, *arg_types)
            func = functype(func_ptr.value)
            func(self.ptr, ctypes.c_void_p(self.hwnd), *args)
        except Exception:
            pass

    def set_progress(self, current, total):
        self._call_method(9, int(current), int(total))  # SetProgressValue

    def set_state(self, state_flag):
        self._call_method(10, state_flag)  # SetProgressState


_ctrl: _TaskbarController | None = None


def init_taskbar(widget) -> None:
    """Initialize the controller for a shown top-level Qt widget/window."""
    global _ctrl
    if sys.platform != "win32" or _ctrl is not None:
        return
    try:
        c = _TaskbarController()
        win = widget.window()
        hwnd = int(win.winId())
        c.set_hwnd(hwnd)
        _ctrl = c
    except Exception:
        _ctrl = None


def set_taskbar_progress(progress: float, state: str = "normal") -> None:
    """progress: 0.0-1.0. state: normal(green)/paused(yellow)/error(red)/indeterminate/none."""
    if _ctrl is None:
        return
    try:
        flag = {"none": TBPF_NOPROGRESS, "indeterminate": TBPF_INDETERMINATE,
                "normal": TBPF_NORMAL, "error": TBPF_ERROR, "paused": TBPF_PAUSED}.get(state, TBPF_NORMAL)
        _ctrl.set_state(flag)
        if state not in ("none", "indeterminate"):
            _ctrl.set_progress(int(progress * 1000), 1000)
    except Exception:
        pass
