"""
qr_code.py — QR code generation for sharing the gallery URL on phone.

Tries `qrcode` library first (better quality), falls back to a minimal
ASCII text representation if not installed.
"""
import os


def generate_qr_image(url: str, output_path: str, box_size: int = 10) -> bool:
    """
    Generate a PNG QR code image. Returns True on success.

    Requires the `qrcode` library (pip install qrcode[pil]).
    """
    try:
        import qrcode
    except ImportError:
        return False
    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        return True
    except Exception as e:
        print(f"[QR] Erreur: {e}")
        return False


def generate_qr_terminal(url: str) -> str:
    """
    Generate an ASCII QR code as a string (for terminal display).
    Useful as fallback when no PIL/qrcode is available.
    """
    try:
        import qrcode
    except ImportError:
        return f"QR code indisponible (pip install qrcode).\nURL: {url}"
    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=1,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        from io import StringIO
        buf = StringIO()
        qr.print_ascii(out=buf, invert=True)
        return buf.getvalue()
    except Exception as e:
        return f"QR code erreur: {e}\nURL: {url}"


def is_qrcode_available() -> bool:
    """Check if qrcode library is available."""
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        return False
