import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
# LIA lane: the byte-identical template66 copy is APPENDED (not prepended) so that other lanes'
# `import gems57` keeps resolving to their own vendored copy (vendor/gems57) first.
_t66 = str(ROOT / "vendor" / "template66")
if _t66 not in sys.path:
    sys.path.append(_t66)
