"""Check every migrated source/receipt hash without loading models or networks."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MIGRATION_MANIFEST.json").read_text())
checked = 0
for record in manifest["files"]:
    path = root / record["destination"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["destination_sha256"]:
        raise SystemExit(f"snapshot mismatch: {record['destination']}")
    checked += 1
print(json.dumps({"status": "PASS", "checked_files": checked,
                  "snapshot_id": manifest["snapshot_id"]}))
