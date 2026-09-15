# Offline Collection and Deferred Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a network-independent collection and live-labeling workflow that writes `labels.json` locally, then uploads the dataset and merges labels into a registered remote task after connectivity returns.

**Architecture:** New collections are local-first records driven by an offline scanner thread. Label persistence is isolated in atomic local-manifest helpers, while a separate one-shot upload worker binds a finished local session to a remote task and performs rsync plus the existing locked remote manifest merge. Existing historical records and remote conversion behavior remain readable.

**Tech Stack:** Python 3, Flask, `unittest`, OpenSSH, rsync, vanilla JavaScript, HTML/CSS.

**Spec:** `docs/superpowers/specs/2026-08-19-offline-collection-upload-design.md`

## Global Constraints

- Starting, scanning, marking, and stopping a new offline collection must not invoke SSH, rsync, remote directory creation, or Host testing.
- Every label change must be atomically persisted to `<local_dir>/labels.json`.
- A repeated scan must preserve labels by stable relative-path identity.
- Deferred upload must include raw data and `labels.json`, then atomically merge the legacy task manifest at the remote task root.
- Upload failures must preserve all local data and remain retryable.
- Existing SSH commands must keep `ClearAllForwardings=yes`.
- Existing historical `collected` and `imported` records remain readable; remote conversion behavior is out of scope.

---

### Task 1: Test isolation and atomic local label manifest

**Files:**
- Create: `tests/test_offline_collections.py`
- Modify: `app.py` near `episode_number`, `write_manifest`, and `refresh_collection_files`

**Interfaces:**
- Produces: `labels_path(collection: dict[str, Any]) -> Path`
- Produces: `write_local_labels(collection: dict[str, Any]) -> str`
- Produces: `load_local_label_statuses(local_root: Path) -> dict[str, str]`
- Produces: `refresh_collection_files(item: dict[str, Any]) -> None` preserving status by relative path

- [ ] **Step 1: Add a temporary-state test fixture**

Patch module globals during each test so tests never touch the real `state/studio.json`:

```python
class OfflineCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_state_file = app.STATE_FILE
        self.old_store = app.store
        app.STATE_FILE = self.root / "state.json"
        app.store = app.Store()

    def tearDown(self):
        app.store = self.old_store
        app.STATE_FILE = self.old_state_file
        self.tmp.cleanup()
```

- [ ] **Step 2: Write failing tests for local JSON location and schema**

Create a collection with valid, invalid, and unreviewed episodes, call `write_local_labels`, and assert:

```python
path = Path(app.write_local_labels(collection))
self.assertEqual(path, local_dir / "labels.json")
payload = json.loads(path.read_text("utf-8"))
self.assertEqual(payload["version"], 1)
self.assertEqual(payload["session"], local_dir.name)
self.assertEqual(payload["counts"], {"valid": 1, "invalid": 1, "unreviewed": 1})
self.assertEqual(payload["valid"][0]["relative_path"], "episode_000001.mp4")
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_offline_collections.OfflineCollectionTests.test_local_labels -v
```

Expected: FAIL because `write_local_labels` does not exist.

- [ ] **Step 4: Implement atomic local manifest helpers**

Add helpers that build the versioned payload and write a temporary file in the same directory:

```python
def labels_path(collection: dict[str, Any]) -> Path:
    return Path(collection["local_dir"]) / "labels.json"


def write_local_labels(collection: dict[str, Any]) -> str:
    path = labels_path(collection)
    groups = {status: [] for status in ("valid", "invalid", "unreviewed")}
    for episode in collection.get("episodes", []):
        status = episode.get("status", "unreviewed")
        groups[status].append({
            "name": episode["name"],
            "relative_path": episode["relative_path"],
            "episode_number": episode_number(episode),
        })
    payload = {
        "version": 1,
        "session": Path(collection["local_dir"]).name,
        "created_at": collection["created_at"],
        "updated_at": now_iso(),
        **groups,
        "counts": {key: len(value) for key, value in groups.items()},
    }
    fd, tmp_name = tempfile.mkstemp(prefix=".labels-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as dst:
            json.dump(payload, dst, ensure_ascii=False, indent=2)
            dst.write("\n")
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    collection["labels_file"] = str(path)
    return str(path)
```

Import `tempfile` at module top. Validate unsupported status before indexing `groups`.

- [ ] **Step 5: Write failing tests for label preservation and restoration**

Test both in-memory preservation and recovery from `labels.json`:

```python
app.refresh_collection_files(collection)
target = next(e for e in collection["episodes"] if e["relative_path"] == rel)
target["status"] = "valid"
app.write_local_labels(collection)
(local_dir / "episode_000002.mp4").write_bytes(b"video")
collection["episodes"] = []
app.refresh_collection_files(collection)
statuses = {e["relative_path"]: e["status"] for e in collection["episodes"]}
self.assertEqual(statuses[rel], "valid")
self.assertEqual(statuses["episode_000002.mp4"], "unreviewed")
```

- [ ] **Step 6: Run preservation test and verify RED**

Run the new test alone. Expected: FAIL because refresh currently recreates every status as `unreviewed`.

- [ ] **Step 7: Preserve stable identities during scanning**

Before calling `scan_episodes`, combine existing statuses and `load_local_label_statuses(root)`, keyed by `relative_path`. Apply the mapping to new episodes. Exclude root `labels.json`, `.humanoid_data`, and `.labels-*` temporary files from file count, byte count, and episode candidates.

- [ ] **Step 8: Run Task 1 tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_offline_collections -v
```

Expected: all Task 1 tests PASS.

- [ ] **Step 9: Commit Task 1 if Git is available**

```bash
git add app.py tests/test_offline_collections.py
git commit -m "feat: persist offline labels locally"
```

If the directory is still not a Git worktree, record that fact and continue without inventing a repository.

---

### Task 2: Offline scanner state machine and API

**Files:**
- Modify: `tests/test_offline_collections.py`
- Modify: `app.py` near `Store.__init__`, workers, collection endpoints

**Interfaces:**
- Consumes: `refresh_collection_files`, `write_local_labels`
- Produces: `offline_scan_worker(item_id: str, stop: threading.Event) -> None`
- Changes: `POST /api/collections/start` accepts only `local_dir`
- Changes: `POST /api/collections/<id>/stop` completes offline scan without network

- [ ] **Step 1: Write a failing API test proving start has no remote side effects**

Use `unittest.mock.patch` to make forbidden operations raise immediately:

```python
with patch.object(app, "remote_mkdir", side_effect=AssertionError("network used")), \
     patch.object(app, "run_command", side_effect=AssertionError("network used")), \
     patch.object(app.subprocess, "Popen", side_effect=AssertionError("network used")):
    response = app.app.test_client().post(
        "/api/collections/start", json={"local_dir": str(local_dir)}
    )
self.assertEqual(response.status_code, 200)
self.assertEqual(response.get_json()["collection"]["status"], "collecting_offline")
```

Patch `threading.Thread.start` in this unit test to avoid racing the worker.

- [ ] **Step 2: Run the test and verify RED**

Expected: FAIL because the endpoint requires a task/Host and starts `sync_worker`.

- [ ] **Step 3: Implement offline start and scanner worker**

Create the collection without remote fields:

```python
item = {
    "id": "collect_" + uuid.uuid4().hex[:8],
    "source_type": "offline",
    "local_dir": local_dir,
    "labels_file": str(Path(local_dir) / "labels.json"),
    "status": "collecting_offline",
    "message": "正在扫描本地数采数据",
    "created_at": now_iso(),
    "file_count": 0,
    "bytes_total": 0,
    "episodes": [],
    "logs": [],
}
```

`offline_scan_worker` loops until `stop.wait(3)` returns true, persists state after each scan, then performs one final scan plus `write_local_labels` and sets `pending_upload`. Exceptions set `interrupted` for scanning failures while retaining local paths and logs.

- [ ] **Step 4: Write a failing stop-worker test**

Run `offline_scan_worker` with an already-set event and assert final scan, `labels.json`, and `pending_upload`:

```python
stop = threading.Event()
stop.set()
app.offline_scan_worker(item["id"], stop)
self.assertEqual(item["status"], "pending_upload")
self.assertTrue((local_dir / "labels.json").is_file())
```

- [ ] **Step 5: Implement offline stop endpoint behavior**

For `source_type == "offline"`, set the event and status `stopping_offline`. If no live worker exists (for an interrupted process), synchronously perform the final scan/write and set `pending_upload`, allowing recovery after restart. Preserve legacy handling only for historical source types.

- [ ] **Step 6: Update restart migration**

In `Store.__init__`, map `collecting_offline`, `stopping_offline`, and `uploading` to `interrupted` with a recovery message. Keep existing legacy mappings.

- [ ] **Step 7: Run Task 2 tests**

Run the entire `tests.test_offline_collections` module and expect PASS.

- [ ] **Step 8: Commit Task 2 if Git is available**

```bash
git add app.py tests/test_offline_collections.py
git commit -m "feat: add offline collection scanner"
```

---

### Task 3: Make marking and preview strictly local

**Files:**
- Modify: `tests/test_offline_collections.py`
- Modify: `app.py` near mark, refresh, and manifest endpoints

**Interfaces:**
- Consumes: `write_local_labels`, `load_local_label_statuses`
- Changes: mark and refresh persist `labels.json` without SSH
- Changes: manifest preview returns local `labels.json`

- [ ] **Step 1: Write a failing mark endpoint test**

Insert an offline collection, patch all remote helpers to fail if called, mark an episode, then assert the response succeeds and root `labels.json` contains the mark.

```python
response = client.post(
    f"/api/collections/{item['id']}/mark",
    json={"episode_id": episode["id"], "status": "valid"},
)
self.assertEqual(response.status_code, 200)
self.assertEqual(json.loads((local_dir / "labels.json").read_text())["counts"]["valid"], 1)
```

- [ ] **Step 2: Run it and verify RED**

Expected: FAIL because `mark_episode` calls `write_manifest(upload=True)` and requires remote task fields.

- [ ] **Step 3: Implement local-only marking with rollback on write failure**

For offline collections, store the old status, set the new one, attempt `write_local_labels`, and restore the old status if persistence fails. Save state only after a successful local write. Historical records retain their existing behavior.

- [ ] **Step 4: Add refresh and preview tests**

Assert `/refresh` preserves statuses and rewrites `labels.json`; assert `/manifest` returns filename `labels.json`, its absolute path, and the versioned local payload.

- [ ] **Step 5: Implement offline branches in refresh and preview**

`refresh` calls local scanning and `write_local_labels`. `get_manifest` reads or creates `labels.json`; it must never call task lookup or SSH for offline sessions.

- [ ] **Step 6: Run all backend tests**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all existing SSH tests and new local-flow tests PASS.

- [ ] **Step 7: Commit Task 3 if Git is available**

```bash
git add app.py tests/test_offline_collections.py
git commit -m "feat: keep live labeling fully offline"
```

---

### Task 4: Deferred upload worker and retryable endpoint

**Files:**
- Modify: `tests/test_offline_collections.py`
- Modify: `app.py` near remote manifest helpers and collection endpoints

**Interfaces:**
- Produces: `remote_manifest_entry(collection: dict[str, Any]) -> dict[str, Any]`
- Produces: `merge_remote_manifest(collection: dict[str, Any]) -> str`
- Produces: `deferred_upload_worker(item_id: str) -> None`
- Produces: `POST /api/collections/<id>/upload`

- [ ] **Step 1: Write failing validation tests for upload request**

Test rejection when the collection is active, target path is outside the task root, another worker is alive, or a labeled episode lacks a numeric suffix. Test acceptance for `pending_upload` and `upload_error`.

- [ ] **Step 2: Run validation tests and verify RED**

Expected: 404 because the upload endpoint does not exist.

- [ ] **Step 3: Extract legacy remote entry construction**

Move the existing numeric list construction into:

```python
def remote_manifest_entry(collection: dict[str, Any]) -> dict[str, Any]:
    labeled = [e for e in collection.get("episodes", []) if e["status"] in {"valid", "invalid"}]
    missing = [e["name"] for e in labeled if episode_number(e) is None]
    if missing:
        raise ValueError("以下已标注 Episode 无数字后缀，无法写入远端任务 JSON: " + ", ".join(missing))
    valid = sorted(episode_number(e) for e in labeled if e["status"] == "valid")
    invalid = sorted(episode_number(e) for e in labeled if e["status"] == "invalid")
    return {"valid_count": len(valid), "valid": valid, "invalid": invalid}
```

Reuse it from the old manifest path and new upload path.

- [ ] **Step 4: Implement upload endpoint validation and thread startup**

Validate collection status, task, Host, and descendant remote path. Persist `task_id`, `task_name`, `host_id`, and `remote_dir`, set `uploading`, then register a worker of kind `deferred_upload` before starting its daemon thread.

- [ ] **Step 5: Write a failing worker command test**

Mock `remote_mkdir`, `run_command`, and remote merge. Assert rsync receives the root directory with no `labels.json` exclusion:

```python
self.assertIn(str(local_dir) + "/", rsync_args)
self.assertNotIn("labels.json", " ".join(rsync_args))
self.assertIn("--exclude", rsync_args)
self.assertIn(".humanoid_data/", rsync_args)
```

Assert success sets `uploaded`, `uploaded_at`, `last_sync_at`, and remote manifest path.

- [ ] **Step 6: Implement one-shot deferred upload**

Worker sequence:

1. Resolve collection, task, and Host.
2. Ensure local `labels.json` exists by writing it.
3. Create remote session directory.
4. Run `rsync -az --partial` with Studio's `rsync_ssh()`, excluding only `.humanoid_data/`.
5. Call locked `REMOTE_MANIFEST_MERGE` with `remote_manifest_entry`.
6. Set `uploaded` only after both remote operations return successfully.
7. On any exception, set `upload_error`, log the error, and retain binding fields for retry.
8. Always remove the worker registry entry.

- [ ] **Step 7: Add failure and retry tests**

First mock rsync failure and assert `upload_error`; then invoke upload again with success mocks and assert `uploaded`. Separately test remote merge failure after successful rsync.

- [ ] **Step 8: Run backend suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile app.py
```

Expected: zero failures and syntax exit code 0.

- [ ] **Step 9: Commit Task 4 if Git is available**

```bash
git add app.py tests/test_offline_collections.py
git commit -m "feat: upload completed offline sessions"
```

---

### Task 5: Offline-first collection UI and upload dialog

**Files:**
- Modify: `static/index.html`
- Modify: `static/app.js`
- Modify: `static/style.css` only if existing layout classes cannot express the dialog/action layout
- Create: `tests/test_frontend_contract.py`

**Interfaces:**
- Consumes: new offline start/stop/upload endpoints and statuses
- Produces: `openUploadModal(id)`, `fillUploadDefaults()`, `uploadCollection()`

- [ ] **Step 1: Write failing frontend contract tests**

Read static files as text and assert required controls/copy exist while realtime-upload copy is gone:

```python
self.assertIn('id="localDir"', html)
self.assertIn('开始离线数采', html)
self.assertIn('id="uploadModal"', html)
self.assertIn('id="uploadTask"', html)
self.assertIn('function uploadCollection()', js)
self.assertNotIn('开始数采与实时上传', html)
self.assertNotIn('toast(\'数采实时同步已启动\')', js)
```

- [ ] **Step 2: Run contract tests and verify RED**

Expected: FAIL on missing offline copy and upload dialog.

- [ ] **Step 3: Restructure the collection page**

Make the primary panel “开始离线数采” with only `localDir`. Move the existing remote task panel after it and label it “联网后注册远端任务”. Remove task/Host/remote inputs from offline start. Keep Host settings as an explicit header action.

- [ ] **Step 4: Add deferred upload dialog**

Add `uploadModal` with collection display, `uploadTask`, `uploadHost`, and `uploadRemote`. Task selection fills the default Host and `<task.remote_dir>/<local basename>`.

- [ ] **Step 5: Update status rendering and actions**

Map the new statuses to Chinese labels. Render actions as follows:

- `collecting_offline`: “去标注” and “结束数采”.
- `stopping_offline`: disabled progress state.
- `pending_upload`: “去标注” and “上传”.
- `uploading`: “去标注” plus upload progress.
- `upload_error`: “去标注” and “重试上传”.
- `uploaded`: “去标注” and remote destination text.
- `interrupted`: “去标注”, “结束并生成 JSON”, and upload only after finalization.

- [ ] **Step 6: Preserve review selection during polling**

When `loadState()` refreshes, update `review` from the matching collection if the review page has a selected ID, preserve `currentEp.id`, rerender, and only select a fallback if the current episode disappeared. This allows newly scanned episodes to appear without disrupting the active video.

- [ ] **Step 7: Update notifications and metrics**

Use “离线数采已启动”, “本地 labels.json 已更新”, “正在结束并生成本地 JSON”, and “补传已开始”. Ensure the data metric says “本地已扫描”.

- [ ] **Step 8: Run frontend and backend tests**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 5 if Git is available**

```bash
git add static/index.html static/app.js static/style.css tests/test_frontend_contract.py
git commit -m "feat: add offline-first collection workflow UI"
```

---

### Task 6: Documentation, compatibility check, and end-to-end verification

**Files:**
- Modify: `README.md`
- Verify: `state/studio.json` is not manually rewritten

**Interfaces:**
- Documents the complete offline/online operator workflow

- [ ] **Step 1: Update README workflow**

Document:

```text
1. Connect XSENS_5g and start Studio.
2. Enter only the local session directory and start offline collection.
3. Review and mark episodes while files arrive.
4. End collection and verify <session>/labels.json.
5. Restore internet, register the remote task, and upload the pending session.
6. Verify raw files plus labels.json in the remote session and the task manifest at the task root.
```

Also state explicitly that offline endpoints do not need a reachable Host.

- [ ] **Step 2: Run the complete automated verification**

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile app.py
```

Expected: zero failures and both commands exit 0.

- [ ] **Step 3: Run a local no-network smoke test**

Start Studio on a temporary port and use HTTP requests to:

1. Start a collection in a temporary local directory.
2. Add representative episode files.
3. Wait for a scan and fetch the collection.
4. Mark one valid and one invalid.
5. Stop and poll until `pending_upload`.
6. Assert `<temp session>/labels.json` contains the expected paths and counts.

Do not invoke the upload endpoint in this no-network smoke test.

- [ ] **Step 4: Run an online cluster_0 smoke test with a disposable remote path**

Only after confirming the exact disposable remote target, create a small temporary local session with `labels.json`, register or select a test task, trigger deferred upload, and verify through SSH that:

- remote session contains the source file and `labels.json`;
- remote task root contains the expected task manifest entry;
- Studio status is `uploaded`.

Do not delete any pre-existing remote data. Remove only disposable test files whose exact paths were created by this smoke test, and report what was removed.

- [ ] **Step 5: Inspect final diff and application process state**

Confirm only scoped files changed. If the running server predates the edits, restart it or tell the user exactly how to restart; do not claim the browser is using new code while an old process remains.

- [ ] **Step 6: Commit Task 6 if Git is available**

```bash
git add README.md
git commit -m "docs: describe offline collection and deferred upload"
```

- [ ] **Step 7: Final verification-before-completion gate**

Re-run the full test suite and syntax check fresh, record the output, and summarize the implemented workflow, local JSON path, remote destinations, tests, and any online smoke test not performed.
