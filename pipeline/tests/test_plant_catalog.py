"""Durable catalog progress without external tools, GPU or user data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plant_catalog import CatalogJob
import run_pipeline as runner
from test_run_pipeline import make_config


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def job(self):
        with patch('plant_catalog.git_value', return_value=None):
            return CatalogJob(self.root, 'test-root', 'capture-a', self.root,
                title='test', job_type='reconstruction', command='test', config={})

    def read(self):
        files = list((self.root / 'catalog/jobs').glob('*.json'))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def test_atomic_update_failure_preserves_previous_record(self):
        job = self.job()
        before = self.read()
        with patch('plant_catalog.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                job.milestone('config_ok')
        self.assertEqual(self.read(), before)
        self.assertEqual(list((self.root / 'catalog/jobs').glob('*.tmp')), [])

    def test_progress_survives_without_finalization(self):
        job = self.job()
        job.milestone('config_ok')
        job.start_stage('colmap')
        record = self.read()
        self.assertEqual(record['execution_status'], 'running')
        self.assertEqual(record['progress']['furthest_milestone'], 'config_ok')
        self.assertEqual(record['progress']['active_stage'], 'colmap')
        self.assertIsNone(record['ended_at'])

    def test_finish_tracks_failed_stage_and_elapsed_time(self):
        job = self.job()
        job.start_stage('train')
        job.finish(1, 'training failed')
        record = self.read()
        self.assertEqual(record['execution_status'], 'failed')
        self.assertEqual(record['progress']['stages']['train']['status'], 'failed')
        self.assertIn('training failed', record['error_summary'])
        self.assertGreaterEqual(record['progress']['stages']['train']['elapsed_seconds'], 0)
        self.assertIsNone(record['progress']['active_stage'])

    def invoke(self, config, *extra):
        with patch('plant_catalog.git_value', return_value=None), patch('plant_catalog.subprocess.run', side_effect=FileNotFoundError):
            return runner.main(['--config', str(config), '--catalog-root', str(self.root),
                '--catalog-root-id', 'test-root', '--catalog-capture-id', 'capture-a', *extra])

    def test_invalid_config_is_recorded_without_fabricated_capture(self):
        self.assertEqual(self.invoke(self.root / 'absent.ini'), 2)
        record = self.read()
        self.assertEqual(record['execution_status'], 'failed')
        self.assertEqual(record['capture_id'], 'capture-a')
        self.assertIsNone(record['selected_images'])
        self.assertEqual(record['progress']['milestones'], [])
        self.assertFalse((self.root / 'catalog/captures').exists())

    def test_dry_run_creates_no_catalog(self):
        with patch.object(runner, 'run_pipeline', return_value=0):
            cfg = make_config(self.root)
            self.assertEqual(self.invoke(cfg.source, '--dry-run'), 0)
        self.assertFalse((self.root / 'catalog').exists())

    def test_prerequisite_failure_preserves_config_boundary(self):
        cfg = make_config(self.root)
        (cfg.images_path / 'one.jpg').write_bytes(b'image')
        with patch.object(runner, 'check_prerequisites', side_effect=runner.StageError('missing tool')):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), 1)
        record = self.read()
        self.assertEqual(record['progress']['furthest_milestone'], 'config_ok')
        self.assertIn('missing tool', record['error_summary'])
        self.assertTrue(record['evidence'])

    def test_stage_failure_and_interrupt(self):
        for exception, code, status in [(runner.StageError('bad solve'), 1, 'failed'),
                                         (KeyboardInterrupt(), 130, 'cancelled')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                cfg = make_config(self.root)
                (cfg.images_path / 'one.jpg').write_bytes(b'image')
                with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=exception):
                    self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), code)
                record = self.read()
                self.assertEqual(record['execution_status'], status)
                self.assertEqual(record['progress']['furthest_milestone'], 'prereqs_ok')
                self.assertEqual(record['progress']['stages']['colmap']['status'], status)

    def test_success_verifies_before_marking_milestone(self):
        cfg = make_config(self.root)
        (cfg.images_path / 'one.jpg').write_bytes(b'image')
        def solve(*args):
            sparse = cfg.colmap_workspace(cfg.make_run_id()) / 'sparse/0'
            sparse.mkdir(parents=True)
            (sparse / 'cameras.bin').write_bytes(b'camera')
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=solve):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), 0)
        record = self.read()
        self.assertEqual(record['execution_status'], 'completed')
        self.assertEqual(record['progress']['furthest_milestone'], 'colmap_done')
        self.assertEqual(record['progress']['stages']['colmap']['status'], 'completed')
        self.assertTrue(record['progress']['resolved']['run_id'])
        self.assertTrue(record['output_artifact_ids'])
        self.assertIn('max_num_iterations', record['progress']['resolved']['config'])
        self.assertIn('python', record['progress']['dependencies'])

    def test_successful_command_without_output_is_not_a_verified_stage(self):
        cfg = make_config(self.root)
        (cfg.images_path / 'one.jpg').write_bytes(b'image')
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command'):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), 1)
        record = self.read()
        self.assertEqual(record['progress']['furthest_milestone'], 'prereqs_ok')
        self.assertEqual(record['progress']['stages']['colmap']['status'], 'failed')
        self.assertEqual(record['output_artifact_ids'], [])

    def test_repeated_milestone_preserves_first_verified_timestamp(self):
        job = self.job()
        job.milestone('exported')
        original = self.read()['progress']['milestones']
        job.milestone('exported')
        self.assertEqual(self.read()['progress']['milestones'], original)

    def prepare_workspace(self, **overrides):
        cfg = make_config(self.root, **overrides)
        (cfg.images_path / 'one.jpg').write_bytes(b'image')
        workspace = cfg.colmap_workspace(cfg.make_run_id())
        workspace.mkdir(parents=True)
        return cfg, workspace

    def test_partial_pyramid_preserves_transforms_boundary(self):
        cfg, workspace = self.prepare_workspace()
        def process(*args):
            (workspace / 'transforms.json').write_text(json.dumps({'frames': [{'file_path': 'images/one.jpg'}]}))
            (workspace / 'images').mkdir()
            (workspace / 'images/one.jpg').write_bytes(b'image')
            # One real partial level: existence alone must not count as complete.
            (workspace / 'images_2').mkdir()
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=process):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'process'), 1)
        record = self.read()
        self.assertEqual(record['progress']['furthest_milestone'], 'transforms_ok')
        self.assertNotIn('pyramid_ok', [m['name'] for m in record['progress']['milestones']])
        self.assertIn('missing 1 of 1 registered frames', record['error_summary'])
        self.assertEqual(record['progress']['stages']['process']['status'], 'failed')

    def test_process_then_training_failure_preserves_verified_progress(self):
        cfg, workspace = self.prepare_workspace(train_extra='downscale_factor = 1\neval_mode = all')
        def command(argv, label, dry_run):
            if label == 'ns-process-data':
                (workspace / 'transforms.json').write_text(json.dumps({'frames': [{'file_path': 'images/one.jpg'}]}))
                (workspace / 'images').mkdir()
                (workspace / 'images/one.jpg').write_bytes(b'image')
            else:
                raise runner.StageError('training failed')
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=command):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'process,train', '--allow-full-resolution'), 1)
        record = self.read()
        self.assertEqual(record['progress']['furthest_milestone'], 'pyramid_ok')
        self.assertEqual(record['progress']['stages']['process']['status'], 'completed')
        self.assertEqual(record['progress']['stages']['train']['status'], 'failed')
        self.assertGreaterEqual(record['progress']['stages']['train']['elapsed_seconds'], 0)

    def test_export_success_and_invalid_output_boundaries(self):
        for valid in (False, True):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                cfg, workspace = self.prepare_workspace(train_extra='downscale_factor = 1')
                run_dir = cfg.output_dir(cfg.make_run_id()) / cfg.train_method / '2026-01-01_000000'
                checkpoints = run_dir / 'nerfstudio_models'
                checkpoints.mkdir(parents=True)
                checkpoint = checkpoints / 'step-000000499.ckpt'
                checkpoint.write_bytes(b'checkpoint')
                (run_dir / 'config.yml').write_text('config')
                def export(argv, label, dry_run):
                    cfg.exports_dir.mkdir(parents=True, exist_ok=True)
                    filename = runner.export_filename(cfg, cfg.make_run_id(), 1, cfg.max_num_iterations, None)
                    header = 'ply\nformat ascii 1.0\nelement vertex ' + ('1' if valid else '0') + '\nend_header\n'
                    (cfg.exports_dir / filename).write_text(header + ' ' * 2048)
                with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=export), patch.object(runner, 'read_run_eval_token', return_value=None):
                    self.assertEqual(self.invoke(cfg.source, '--stage', 'export'), 0 if valid else 1)
                record = self.read()
                self.assertEqual(record['progress']['furthest_milestone'], 'verified' if valid else 'exported')
                self.assertEqual(record['execution_status'], 'completed' if valid else 'failed')
                self.assertEqual(record['progress']['stages']['export']['status'], 'completed' if valid else 'failed')
                self.assertEqual(len(record['output_artifact_ids']), 1 if valid else 0)
                self.assertEqual('verified' in [m['name'] for m in record['progress']['milestones']], valid)
