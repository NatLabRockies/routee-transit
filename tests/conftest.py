import sys
from pathlib import Path

# Allow tests to import scripts that are not installed packages.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "gtfs_realtime"))
