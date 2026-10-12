"""Download the schema (EMS) vectors from Zhang et al. (2026) and verify them.

    uv run scripts/download_data.py                          # download from Zenodo
    uv run scripts/download_data.py --archive path/to.zip    # or use a ZIP you downloaded by hand

Data: https://doi.org/10.5281/zenodo.21715792 (v1.0.0, CC BY 4.0). Zenodo sometimes blocks automated downloads; if
so, download the ZIP in a browser from the link above and pass it with --archive. Only the activation vectors, the EMS
definitions and the per-schema judge prompts (trait_data_eval) are unpacked, into data/.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

RECORD = "21715792"
ARCHIVE_NAME = "Mechanistic_Control_LLM_Simulated_Participants_Data_v1.0.0.zip"
MD5 = "6dbbfcfa01f970fbaaa4de240bac237a"          # v1.0.0, as published on Zenodo
ROOT = "mechanistic_control_llm_data_v1_0_0/"
KEEP = [ROOT + "vectors/persona_vector.tar.gz",
        ROOT + "experimental_materials/ems_definitions_used_for_generation.csv",
        ROOT + "LICENSE.txt", ROOT + "README.md"]
KEEP_DIRS = [ROOT + "experimental_materials/trait_data_eval/"]   # per-schema judge prompts and held-out questions
DATA = Path(__file__).resolve().parent.parent / "data"


def md5(path: Path) -> str:
    h = hashlib.md5()                              # integrity check only
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> Path:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}") as r:
        record = json.load(r)
    url = next(f["links"]["self"] for f in record["files"] if f["key"] == ARCHIVE_NAME)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / ARCHIVE_NAME
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="a ZIP downloaded by hand from https://doi.org/10.5281/zenodo.21715792")
    args = ap.parse_args()
    archive = args.archive or download(DATA / ".download")
    got = md5(archive)
    if got != MD5:
        raise SystemExit(f"checksum mismatch: expected {MD5}, got {got}. Is this v1.0.0 of the dataset?")
    print(f"checksum OK ({MD5})")
    out = DATA / "ems"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in KEEP:
            target = out / Path(name).name
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        for folder in KEEP_DIRS:
            target_dir = out / Path(folder).name
            target_dir.mkdir(exist_ok=True)
            for name in z.namelist():
                if name.startswith(folder) and not name.endswith("/"):
                    with z.open(name) as src, (target_dir / Path(name).name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    with tarfile.open(out / "persona_vector.tar.gz") as t:
        t.extractall(out / "vectors", filter="data")
    (out / "persona_vector.tar.gz").unlink()
    vectors = sorted((out / "vectors").rglob("*.pt"))
    print(f"unpacked {len(vectors)} vector files into {out / 'vectors'}")


if __name__ == "__main__":
    main()
