"""Register job folders processed elsewhere (e.g. on Colab by portalcut.batch) in the local app.

    uv run python -m portalcut.import_job ~/Google\\ Drive/My\\ Drive/portalcut/jobs/<job_id> [...]

The folder is copied into data/<job_id> (skip with --move to move instead) and its job.json is added
to the app's database, so it shows up on the page and dashboard like a job processed here.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from portalcut import db


def import_job(folder: Path, move: bool = False, replace: bool = False) -> str:
    folder = Path(folder)
    job = json.loads((folder / "job.json").read_text())
    jid = job["id"]
    if not jid.isalnum():  # the id becomes a folder name in data/
        raise ValueError(f"bad job id {jid!r}")
    db.init()
    if db.get(jid) is not None:
        if not replace:
            raise FileExistsError(f"job {jid} is already in the app (use --replace)")
        db.delete(jid)
    dest = db.DATA_DIR / jid
    if dest.resolve() != folder.resolve():
        if dest.exists():
            shutil.rmtree(dest)
        (shutil.move if move else shutil.copytree)(str(folder), str(dest))
    db.create(jid, job["filename"])
    db.update(jid, status=job["status"], progress=job.get("progress", 1.0), error=job.get("error"), meta=job.get("meta"))
    return jid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--replace", action="store_true", help="overwrite a job with the same id")
    a = ap.parse_args(argv)
    for f in a.folders:
        print("imported", import_job(f, a.move, a.replace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
