"""Durable catalog progress without external tools, GPU or user data."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plant_catalog
from plant_catalog import CatalogJob, MILESTONES, read_progress
import run_pipeline as runner
from test_run_pipeline import make_config
import colmap_threads


def setUpModule():
    # These tests run the colmap stage, whose thread choice reads the machine's
    # memory (and Windows', on WSL). Fixed figures keep them hermetic.
    figures = colmap_threads.HostMemory(linux_available_mib=65536, usable_cpus=8)
    patcher = patch.object(colmap_threads, "read_host_memory", return_value=figures)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    version = patch.object(runner, "detect_colmap_version_parts", return_value=(3, 12, 0))
    version.start()
    unittest.addModuleCleanup(version.stop)


class CatalogFixture(unittest.TestCase):
    """Temporary catalog root plus helpers; holds no tests of its own."""

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

    def invoke(self, config, *extra):
        with patch('plant_catalog.git_value', return_value=None), patch('plant_catalog.subprocess.run', side_effect=FileNotFoundError):
            return runner.main(['--config', str(config), '--catalog-root', str(self.root),
                '--catalog-root-id', 'test-root', '--catalog-capture-id', 'capture-a', *extra])


class CatalogTests(CatalogFixture):
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

    def test_invalid_config_is_recorded_without_fabricated_capture(self):
        self.assertEqual(self.invoke(self.root / 'absent.ini'), 2)
        record = self.read()
        self.assertEqual(record['execution_status'], 'failed')
        self.assertEqual(record['capture_id'], 'capture-a')
        self.assertIsNone(record['selected_images'])
        self.assertEqual(record['progress']['milestones'], [])
        self.assertIsNone(record['progress']['dependencies']['colmap'])
        self.assertEqual(record['progress']['failure'],
                         dict(cause='config', stage=None, after_milestone=None))
        self.assertFalse((self.root / 'catalog/captures').exists())

    def test_catalog_probes_configured_colmap_after_loading_config(self):
        cfg = make_config(self.root)
        text = cfg.source.read_text(encoding="utf-8").replace(
            "[colmap]", "[colmap]\nexecutable = /opt/candidate/bin/colmap")
        cfg.source.write_text(text, encoding="utf-8")
        with patch.object(plant_catalog, "colmap_version", return_value="COLMAP 4.2.1") as probe, \
                patch.object(runner, "run_pipeline", return_value=0):
            self.assertEqual(self.invoke(cfg.source, "--stage", "colmap"), 0)
        probe.assert_called_once_with("/opt/candidate/bin/colmap")
        self.assertEqual(self.read()["progress"]["dependencies"]["colmap"], "COLMAP 4.2.1")

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
        self.assertEqual(record['progress']['failure'],
                         dict(cause='error', stage=None, after_milestone='config_ok'))

    def test_stage_failure_and_interrupt(self):
        for exception, code, status, cause in [(runner.StageError('bad solve'), 1, 'failed', 'error'),
                                                (KeyboardInterrupt(), 130, 'cancelled', 'interrupted')]:
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
                self.assertEqual(record['progress']['failure'],
                                 dict(cause=cause, stage='colmap', after_milestone='prereqs_ok'))

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


class ProgressSchemaTwoTests(CatalogFixture):
    """Schema 2 fields: ladder, failure, GPU, artifacts, SIGTERM, old records."""

    def solve_fixture(self):
        cfg = make_config(self.root)
        (cfg.images_path / 'one.jpg').write_bytes(b'image')
        def solve(*args):
            sparse = cfg.colmap_workspace(cfg.make_run_id()) / 'sparse/0'
            sparse.mkdir(parents=True)
            (sparse / 'cameras.bin').write_bytes(b'camera')
        return cfg, solve

    def test_success_records_ladder_failure_and_artifact_sizes(self):
        cfg, solve = self.solve_fixture()
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=solve):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), 0)
        record = self.read()
        progress = record['progress']
        self.assertEqual(progress['schema_version'], plant_catalog.PROGRESS_SCHEMA_VERSION)
        self.assertIsNone(progress['failure'])
        self.assertIsNone(progress['gpu'])  # no probe can run under the mocked subprocess
        names = [m['name'] for m in progress['milestones']]
        self.assertEqual(names, sorted(names, key=MILESTONES.index))
        self.assertEqual(names, ['config_ok', 'prereqs_ok', 'colmap_done'])
        [entry] = progress['artifacts']
        self.assertEqual(entry['artifact_id'], record['output_artifact_ids'][0])
        self.assertEqual(entry['role'], 'colmap-output')
        artifact = json.loads((self.root / 'catalog/artifacts' / (entry['artifact_id'] + '.json')).read_text())
        self.assertEqual(entry['relative_path'], artifact['location']['relative_path'])
        self.assertIsNone(artifact['sha256'])  # hashing is off by default
        workspace = self.root / entry['relative_path']
        files = [p for p in workspace.rglob('*') if p.is_file()]
        self.assertEqual(entry['file_count'], len(files))
        self.assertEqual(entry['size_bytes'], sum(p.stat().st_size for p in files))
        self.assertEqual(set(progress), {'schema_version', *plant_catalog.PROGRESS_FIELDS})

    def test_sigterm_finalizes_as_terminated_and_restores_handler(self):
        cfg, _ = self.solve_fixture()
        before = signal.getsignal(signal.SIGTERM)
        def killed(*args):
            os.kill(os.getpid(), signal.SIGTERM)
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=killed):
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap'), runner.EXIT_TERMINATED)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        record = self.read()
        self.assertEqual(record['execution_status'], 'cancelled')
        self.assertEqual(record['exit_code'], 143)
        self.assertIn('SIGTERM', record['error_summary'])
        self.assertEqual(record['progress']['stages']['colmap']['status'], 'cancelled')
        self.assertEqual(record['progress']['failure'],
                         dict(cause='terminated', stage='colmap', after_milestone='prereqs_ok'))

    def test_sigterm_handler_only_while_a_catalog_job_is_active(self):
        cfg = make_config(self.root)
        seen = []
        def capture(*args):
            seen.append(signal.getsignal(signal.SIGTERM))
            return 0
        with patch.object(runner, 'run_pipeline', side_effect=capture):
            self.assertEqual(runner.main(['--config', str(cfg.source)]), 0)
        self.assertIsNot(seen[0], runner._raise_terminated)
        self.assertEqual(runner.EXIT_TERMINATED, 143)
        self.assertEqual(plant_catalog.EXIT_CAUSES[runner.EXIT_TERMINATED], 'terminated')

    def test_truncated_write_leaves_previous_record(self):
        job = self.job()
        job.milestone('config_ok')
        before = self.read()
        def partial(record, stream, **kwargs):
            stream.write('{"schema_version": 1, "kind": "jo')
            raise OSError('disk full')
        with patch('plant_catalog.json.dump', side_effect=partial):
            with self.assertRaises(OSError):
                job.milestone('prereqs_ok')
        self.assertEqual(self.read(), before)
        self.assertEqual(list((self.root / 'catalog/jobs').glob('*.tmp')), [])

    def test_ladder_refuses_unknown_names_and_ranks_furthest(self):
        job = self.job()
        before = self.read()
        with self.assertRaises(ValueError):
            job.milestone('nearly_done')
        self.assertEqual(self.read(), before)
        job.milestone('exported')
        job.milestone('config_ok')
        self.assertEqual(self.read()['progress']['furthest_milestone'], 'exported')

    def test_old_records_read_with_explicit_schema_version(self):
        legacy = dict(schema_version=1, kind='job', id='job-legacy')
        version_one = dict(schema_version=1, kind='job', id='job-one', progress=dict(
            schema_version=1, milestones=[dict(name='config_ok', verified_at='t')],
            furthest_milestone='config_ok', active_stage='colmap', stages={}, resolved={}, dependencies={}))
        view = read_progress(legacy)
        self.assertEqual(view['schema_version'], 0)
        self.assertTrue(all(view[field] is None for field in plant_catalog.PROGRESS_FIELDS))
        view = read_progress(version_one)
        self.assertEqual(view['schema_version'], 1)
        self.assertEqual(view['furthest_milestone'], 'config_ok')
        self.assertEqual(view['active_stage'], 'colmap')
        self.assertIsNone(view['gpu'])
        self.assertIsNone(view['artifacts'])
        self.assertIsNone(view['failure'])
        current = self.job().record
        self.assertEqual(read_progress(current)['schema_version'], 2)
        self.assertEqual(read_progress(current)['artifacts'], [])

    def test_consumer_fixtures_keep_the_external_read_contract(self):
        """The producer side of catalog-consumer-compat.cjs, without Mission Control."""
        target = self.root / 'fixtures'
        script = Path(__file__).resolve().parent / 'catalog_consumer_fixtures.py'
        result = subprocess.run([sys.executable, str(script), str(target)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = {path.stem: json.loads(path.read_text()) for path in (target / 'catalog/jobs').glob('*.json')}
        self.assertEqual(len(jobs), 3)
        legacy = jobs.pop('job-legacy')
        self.assertNotIn('progress', legacy)
        completed = [job for job in jobs.values() if job['execution_status'] == 'completed']
        failed = [job for job in jobs.values() if job['execution_status'] == 'failed']
        self.assertEqual(len(completed), 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(completed[0]['progress']['furthest_milestone'], 'colmap_done')
        self.assertIsNone(failed[0]['selected_images'])
        self.assertIsNone(failed[0]['progress']['furthest_milestone'])
        for job in (legacy, *jobs.values()):
            # Catalog v1 fields stay as they were; schema 2 only adds inside progress.
            self.assertEqual(job['schema_version'], 1)
            self.assertEqual(set(legacy) - {'progress'}, set(job) - {'progress'})

    def test_hash_flag_reaches_the_job(self):
        cfg, solve = self.solve_fixture()
        with patch.object(runner, 'check_prerequisites'), patch.object(runner, 'run_command', side_effect=solve), \
                patch.object(runner, 'CatalogJob', wraps=CatalogJob) as made:
            self.assertEqual(self.invoke(cfg.source, '--stage', 'colmap', '--catalog-hash-artifacts'), 0)
        self.assertIs(made.call_args.kwargs['hash_artifacts'], True)


class ArtifactAndProbeTests(unittest.TestCase):
    """Sizes, opt-in hashes and tool probes that degrade to None."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def job(self, **kwargs):
        with patch('plant_catalog.git_value', return_value=None):
            return CatalogJob(self.root, 'test-root', 'capture-a', self.root, title='test',
                              job_type='reconstruction', command='test', config={}, **kwargs)

    def test_hashing_is_opt_in_and_files_only(self):
        output = self.root / 'out'
        (output / 'nested').mkdir(parents=True)
        (output / 'nested/a.bin').write_bytes(b'abc')
        ply = self.root / 'export.ply'
        ply.write_bytes(b'ply data')
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                job = self.job(hash_artifacts=enabled)
                job.artifact(ply, role='point-cloud')
                job.artifact(output, role='train-output')
                file_entry, folder_entry = job.record['progress']['artifacts']
                self.assertEqual((file_entry['size_bytes'], file_entry['file_count']), (8, None))
                self.assertEqual((folder_entry['size_bytes'], folder_entry['file_count']), (3, 1))
                hashes = [json.loads((self.root / 'catalog/artifacts' / (entry['artifact_id'] + '.json')).read_text())['sha256']
                          for entry in (file_entry, folder_entry)]
                self.assertEqual(hashes, [hashlib.sha256(b'ply data').hexdigest() if enabled else None, None])

    def test_unreadable_artifact_size_is_unknown(self):
        self.assertEqual(plant_catalog.measure(self.root / 'absent'), (None, None))

    def test_dependency_probes_degrade_to_unknown(self):
        job = self.job()
        for failure in (FileNotFoundError(), subprocess.TimeoutExpired('colmap', 10), PermissionError()):
            with self.subTest(failure=type(failure).__name__), \
                    patch('plant_catalog.importlib.metadata.version', side_effect=RuntimeError('broken metadata')), \
                    patch('plant_catalog.subprocess.run', side_effect=failure):
                job.dependencies()
                versions = job.record['progress']['dependencies']
                self.assertTrue(versions['python'])
                for name in ('nerfstudio', 'gsplat', 'torch', 'colmap', 'ffmpeg'):
                    self.assertIsNone(versions[name], name)

    def test_colmap_banner_version_when_version_flag_is_rejected(self):
        banner = 'COLMAP 3.12.0 (Commit abc123 on 2025-01-01 without CUDA)\n\nUsage:\n  colmap [command]\n'
        replies = [subprocess.CompletedProcess(['colmap', '--version'], 1, '', 'unrecognised option'),
                   subprocess.CompletedProcess(['colmap'], 0, banner, '')]
        with patch('plant_catalog.subprocess.run', side_effect=replies):
            self.assertEqual(plant_catalog.colmap_version(),
                             'COLMAP 3.12.0 (Commit abc123 on 2025-01-01 without CUDA)')
        with patch('plant_catalog.subprocess.run',
                   return_value=subprocess.CompletedProcess(['ffmpeg'], 1, '', 'error')):
            self.assertIsNone(plant_catalog.ffmpeg_version())

    def test_gpu_inventory_without_nvidia_smi_is_unknown(self):
        with patch('plant_catalog.shutil.which', return_value=None), \
                patch.object(plant_catalog, '_NVIDIA_SMI_WSL', self.root / 'absent'):
            self.assertIsNone(plant_catalog.gpu_inventory())

    def test_gpu_inventory_parses_devices_and_falls_back(self):
        full = subprocess.CompletedProcess([], 0, 'NVIDIA GeForce RTX 3080, 8.6, 10240, 555.42\n', '')
        rejected = subprocess.CompletedProcess([], 6, '', 'Field "compute_cap" is not a valid field')
        older = subprocess.CompletedProcess([], 0, 'Tesla T4, [N/A], 470.1\n', '')
        with patch('plant_catalog.shutil.which', return_value='/usr/bin/nvidia-smi'), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with patch('plant_catalog.subprocess.run', return_value=full):
                self.assertEqual(plant_catalog.gpu_inventory(), dict(cuda_visible_devices='0', devices=[dict(
                    name='NVIDIA GeForce RTX 3080', compute_capability='8.6',
                    memory_total_mib=10240, driver_version='555.42')]))
            with patch('plant_catalog.subprocess.run', side_effect=[rejected, older]):
                self.assertEqual(plant_catalog.gpu_inventory()['devices'], [dict(
                    name='Tesla T4', compute_capability=None, memory_total_mib=None, driver_version='470.1')])
            with patch('plant_catalog.subprocess.run', return_value=rejected):
                self.assertIsNone(plant_catalog.gpu_inventory())

    def test_gpu_probe_error_is_recorded_as_unknown(self):
        job = self.job()
        with patch('plant_catalog.gpu_inventory', side_effect=RuntimeError('driver crashed')):
            job.gpu()
        self.assertIsNone(job.record['progress']['gpu'])
