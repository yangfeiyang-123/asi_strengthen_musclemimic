from __future__ import annotations

import sys
from pathlib import Path

# Import through the repository-qualified namespace.  Adding the shuttlecock
# directory itself made ``src`` collide with the repository's top-level
# ``src`` package whenever the full suite had imported that package first.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
