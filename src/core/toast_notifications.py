"""
toast_notifications.py — Windows 11 native toast notifications.

Uses win11toast if available, falls back to plyer, then to messagebox.
Non-blocking, won't crash if no notification system is available.
"""
import sys
import os


def show_toast(title: str, message: str, duration: str = "short",
               icon_path: str = None, on_click=None, audio: str = "default") -> bool:
    """
    Show a Windows 11 toast notification.

    Args:
        title: Toast title (bold)
        message: Toast body text
        duration: "short" (5s) or "long" (25s)
        icon_path: Path to .png/.ico for the toast icon
        on_click: Optional callback when user clicks the toast
        audio: "default", "silent", or sound name like "IM", "Mail", "Reminder"

    Returns:
        True if toast was shown, False if all backends failed.
    """
    if sys.platform != "win32":
        return False

    # Auto-find app icon if not specified
    if not icon_path:
        _core_dir = os.path.dirname(os.path.abspath(__file__))
        _candidate = os.path.normpath(os.path.join(_core_dir, "..", "..", "assets", "icon.png"))
        if os.path.isfile(_candidate):
            icon_path = _candidate

    # Try win11toast (native Windows 11 / 10 toasts via XML)
    try:
        from win11toast import toast
        kwargs = {
            "title": title,
            "body": message,
            "duration": duration,
            "app_id": "Universal SR Studio",
        }
        if icon_path and os.path.exists(icon_path):
            kwargs["icon"] = {"src": icon_path, "placement": "appLogoOverride"}
        # Always provide on_click; without it win11toast prints the dismissal reason
        kwargs["on_click"] = on_click if on_click else lambda *_: None
        if audio == "silent":
            kwargs["audio"] = {"silent": "true"}
        elif audio != "default":
            kwargs["audio"] = audio
        toast(**kwargs)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Toast] win11toast erreur: {e}")

    # Fallback to plyer (cross-platform but less native)
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Universal SR Studio",
            app_icon=icon_path if icon_path and os.path.exists(icon_path) else None,
            timeout=5 if duration == "short" else 25,
        )
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"[Toast] plyer erreur: {e}")

    # Fallback: PowerShell XML toast (zero extra dependencies, Win 10/11)
    try:
        import subprocess

        def _esc(s: str) -> str:
            return (s.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
                     .replace('"', "&quot;"))

        ps = (
            "$null=[Windows.UI.Notifications.ToastNotificationManager,"
            "Windows.UI.Notifications,ContentType=WindowsRuntime];"
            "$null=[Windows.Data.Xml.Dom.XmlDocument,"
            "Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime];"
            "$xml=New-Object Windows.Data.Xml.Dom.XmlDocument;"
            "$xml.LoadXml('<toast><visual><binding template=\"ToastGeneric\">"
            f"<text>{_esc(title)}</text>"
            f"<text>{_esc(message)}</text>"
            + (f'<image placement="appLogoOverride" src="file:///{icon_path.replace(chr(92), "/")}"/>'
               if icon_path and os.path.isfile(icon_path) else "")
            +
            "</binding></visual></toast>');"
            "$toast=New-Object Windows.UI.Notifications.ToastNotification $xml;"
            "[Windows.UI.Notifications.ToastNotificationManager]"
            "::CreateToastNotifier('Universal SR Studio').Show($toast)"
        )
        subprocess.Popen(
            ["powershell", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True
    except Exception as e:
        print(f"[Toast] PowerShell fallback erreur: {e}")

    return False


def show_training_complete_toast(model_name: str, duration_str: str,
                                   best_psnr: float = None, icon_path: str = None) -> bool:
    """Convenience wrapper for training-complete notifications."""
    body_lines = [f"Modele: {model_name}", f"Duree: {duration_str}"]
    if best_psnr:
        body_lines.append(f"Best PSNR: {best_psnr:.4f} dB")
    return show_toast(
        title="Entrainement Termine",
        message="\n".join(body_lines),
        duration="long",
        icon_path=icon_path,
    )


def show_training_error_toast(model_name: str, error_msg: str = "", icon_path: str = None) -> bool:
    """Convenience wrapper for training-failure notifications."""
    body = f"Modele: {model_name}\n"
    if error_msg:
        body += f"Erreur: {error_msg[:200]}"
    return show_toast(
        title="Entrainement Echoue",
        message=body,
        duration="long",
        icon_path=icon_path,
    )


def show_queue_complete_toast(num_done: int, num_total: int, icon_path: str = None) -> bool:
    """Convenience wrapper for queue completion."""
    return show_toast(
        title="Queue Terminee",
        message=f"{num_done} / {num_total} configurations traitees",
        duration="long",
        icon_path=icon_path,
    )
