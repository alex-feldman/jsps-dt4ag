// Exercise the actual external reader without modifying or launching its app.
// Usage: node catalog-consumer-compat.cjs MISSION_CONTROL_ROOT FIXTURE_ROOT
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const consumerRoot = path.resolve(process.argv[2]);
const root = path.resolve(process.argv[3]);
const ts = require(require.resolve('typescript', { paths: [consumerRoot] }));
const sourcePath = path.join(consumerRoot, 'lib/plant-catalog.ts');
const compiled = ts.transpileModule(fs.readFileSync(sourcePath, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true }
}).outputText;
const loaded = new Module(sourcePath, module);
loaded.filename = sourcePath;
loaded.paths = Module._nodeModulePaths(consumerRoot);
const nativeRequire = loaded.require.bind(loaded);
loaded.require = name => name === '@/lib/workspace'
  ? { APP_ROOT: root, WORKSPACE_ROOT: root } : nativeRequire(name);
loaded._compile(compiled, sourcePath);
const { validateCatalogRecord, loadCatalog } = loaded.exports;
process.env.PLANT_CATALOG_CONFIG = path.join(root, 'consumer-registry.json');
process.env.PLANT_CATALOG_ROOTS = JSON.stringify({ 'test-root': root });
(async () => {
  const result = await loadCatalog();
  assert.deepEqual(result.diagnostics, []);
  assert(result.entries.length >= 4);
  for (const entry of result.entries) {
    assert.equal(entry.availability, 'available', entry.id);
    assert(entry.record, entry.id);
    assert.deepEqual(validateCatalogRecord(entry.record), [], entry.id);
  }
  const jobs = result.entries.filter(e => e.kind === 'job');
  assert.equal(jobs.length, 3);
  const legacy = jobs.find(e => e.id === 'job-legacy');
  assert(legacy && !('progress' in legacy.record));
  assert.deepEqual(legacy.diagnostics, []);
  const current = jobs.find(e => e.id !== 'job-legacy' && e.record.execution_status === 'completed');
  assert.equal(current.record.progress.furthest_milestone, 'colmap_done');
  assert.deepEqual(current.diagnostics, []);
  assert(current.files.every(f => f.availability === 'available'));
  const early = jobs.find(e => e.record.execution_status === 'failed');
  assert.equal(early.record.selected_images, null);
  assert.equal(early.record.progress.furthest_milestone, null);
  assert.deepEqual(early.diagnostics, ['Unresolved capture unresolved-capture.']);
  assert(early.files.some(f => f.label.startsWith('Log ') && f.availability === 'available'));
  // Structural errors must still fail rather than pass because fields are additive.
  assert(validateCatalogRecord({ ...current.record, capture_id: null }).length > 0);
  console.log('PASS: actual consumer validates and loads legacy, progress and early-failure records; unresolved capture stays visible.');
})().catch(error => { console.error(error); process.exitCode = 1; });
