"""
One-off migration for checkpoints trained before se_block_cbam.py was split up.

Ultralytics pickles the whole model, so older .pt files store the attention
class as `se_block_cbam.CBAM`. That module no longer exists (CBAM now lives in
cbam.py), so those checkpoints fail to load. This script maps the old module
name to cbam, loads each checkpoint and saves it back, so it records
`cbam.CBAM` from then on. A <name>.pt.bak copy is kept before rewriting.

Usage (on Colab, where the checkpoints live):
    python main_pipeline/model/migrate_checkpoints.py <file.pt or folder> [...]
"""

import shutil
import sys
import zipfile
from pathlib import Path

# Make cbam.py importable no matter where this script is run from
YOLO_MODS_PATH = Path(__file__).parent.resolve()
sys.path.insert(0, str(YOLO_MODS_PATH))

import torch
import cbam

OLD_MODULE = b"se_block_cbam"


def needs_migration(path):
    """True if the checkpoint's pickle data still references se_block_cbam."""
    try:
        with zipfile.ZipFile(path) as zf:
            return any(
                OLD_MODULE in zf.read(name)
                for name in zf.namelist()
                if name.endswith(".pkl")
            )
    except zipfile.BadZipFile:
        # Legacy (non-zip) torch format: search the raw bytes
        return OLD_MODULE in path.read_bytes()


def migrate(path):
    if not needs_migration(path):
        print(f"skip      {path} (no se_block_cbam reference)")
        return False

    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(ckpt, path)

    if needs_migration(path):
        shutil.copy2(backup, path)
        raise RuntimeError(f"{path} still references se_block_cbam; restored from backup")

    print(f"migrated  {path} (backup: {backup.name})")
    return True


def find_checkpoints(targets):
    for target in map(Path, targets):
        if target.is_dir():
            yield from sorted(target.rglob("*.pt"))
        elif target.suffix == ".pt":
            yield target
        else:
            print(f"ignored   {target} (not a .pt file or folder)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Any pickle that asks for se_block_cbam.CBAM now gets cbam.CBAM
    sys.modules["se_block_cbam"] = cbam

    checkpoints = list(find_checkpoints(sys.argv[1:]))
    migrated = sum(migrate(p) for p in checkpoints)
    print(f"\nDone: {migrated} of {len(checkpoints)} checkpoint(s) migrated")


if __name__ == "__main__":
    main()
