"""Generate real producer records for the optional external consumer check.

Usage: python3 pipeline/tests/catalog_consumer_fixtures.py NEW_DIRECTORY
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_pipeline as runner
from test_run_pipeline import make_config

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
cfg = make_config(root)
(cfg.images_path / 'one.jpg').write_bytes(b'image')

def solve(*args):
    sparse = cfg.colmap_workspace(cfg.make_run_id()) / 'sparse/0'
    sparse.mkdir(parents=True)
    (sparse / 'cameras.bin').write_bytes(b'camera')

def invoke(config, capture):
    with patch('plant_catalog.git_value', return_value=None), patch('plant_catalog.subprocess.run', side_effect=FileNotFoundError):
        return runner.main(['--config', str(config), '--catalog-root', str(root),
            '--catalog-root-id', 'test-root', '--catalog-capture-id', capture, '--stage', 'colmap'])

with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=solve):
    assert invoke(cfg.source, 'capture-a') == 0
jobs = root / 'catalog/jobs'
current = json.loads(next(jobs.glob('*.json')).read_text())
legacy = copy.deepcopy(current)
legacy.pop('progress')
legacy['id'] = 'job-legacy'
legacy['output_artifact_ids'] = []
(jobs / 'job-legacy.json').write_text(json.dumps(legacy))
assert invoke(root / 'missing.ini', 'unresolved-capture') == 2
(root / 'consumer-registry.json').write_text(json.dumps(dict(schema_version=1,
    roots=['test-root'], manifest_directories=[dict(root_id='test-root', relative_path='catalog')], records=[])))
print('Consumer fixtures:', root)
