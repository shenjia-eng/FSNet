"""Verify downloaded release assets against the repository SHA256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', help='Directory containing the release assets')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'release_assets.json').read_text(encoding='utf-8'))
directory = Path(args.directory)
failed = []
for item in manifest['models'] + manifest['datasets']:
    path = directory / item['file']
    valid = path.is_file() and path.stat().st_size == item['bytes']
    if valid:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        valid = digest.hexdigest() == item['sha256']
    print(('OK ' if valid else 'FAIL ') + item['file'])
    if not valid:
        failed.append(item['file'])
raise SystemExit(1 if failed else 0)
