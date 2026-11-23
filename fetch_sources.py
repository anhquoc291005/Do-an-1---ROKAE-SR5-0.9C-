"""Download the exact source revisions used for the SR5 demos."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
for name, source in json.loads((root / 'sources.json').read_text()).items():
    target = root / 'external' / name
    if not target.exists():
        subprocess.run(['git', 'clone', '--no-checkout', source['url'], target], check=True)
        subprocess.run(['git', '-C', target, 'checkout', '--detach', source['commit']], check=True)
    head = subprocess.check_output(['git', '-C', target, 'rev-parse', 'HEAD'], text=True).strip()
    if head != source['commit']:
        raise SystemExit(f'{target}: expected {source["commit"]}, found {head}')
    print(f'{name} {head}')
