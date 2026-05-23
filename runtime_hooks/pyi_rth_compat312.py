# Runtime hook — Python 3.12+ compatibility shim
# pkgutil.ImpImporter was removed in Python 3.12.
# Older pkg_resources (bundled by PyInstaller from system site-packages)
# still references it. Inject a stub before pkg_resources loads.
import pkgutil

if not hasattr(pkgutil, 'ImpImporter'):
    class _ImpImporterStub:
        """Stub for removed pkgutil.ImpImporter (Python 3.12+)."""
        def __init__(self, path=None):
            self.path = path

        def find_module(self, fullname, path=None):
            return None

        def iter_modules(self, prefix=''):
            return iter([])

    pkgutil.ImpImporter = _ImpImporterStub
