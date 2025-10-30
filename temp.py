import hashlib
from pathlib import Path
import shutil
import json
import os

dest = Path("./flat_plates/valid")
manifest_path = Path("./flat_plates/valid/valid_manifest.jsonl")

dest.mkdir(parents=True, exist_ok=True)
manifest_path.parent.mkdir(parents=True, exist_ok=True)

manifest = open(manifest_path, "w", encoding="UTF-8")

for dirpath, dirnames, filenames in os.walk("plates/valid"):
    classification = Path(dirpath).name
    if classification == "valid":
        continue

    for file in filenames:
        src = Path(dirpath) / file
        
        hash_obj = hashlib.md5(str(src).encode())  # md5 hash
        short_hash = hash_obj.hexdigest()[:8]      # take first 8 chars
        dst = dest / f"{short_hash}.jpg"           # new short hashed filename

        obj = {"source-ref": str(dst), "class-label": classification}
        manifest.write(json.dumps(obj) + "\n")

        # Move file
        shutil.move(src, dst)

manifest.close()
