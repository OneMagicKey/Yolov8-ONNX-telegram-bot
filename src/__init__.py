import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))
