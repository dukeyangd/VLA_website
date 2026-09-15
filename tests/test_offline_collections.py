import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class MemoryStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.data = app.default_state()

    def find(self, group, item_id):
        return next((x for x in self.data[group] if x.get("id") == item_id), None)

    def save(self):
        pass


class OfflineCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_store = app.store
        app.store = MemoryStore()
        app.workers.clear()

    def tearDown(self):
        app.workers.clear()
        app.store = self.old_store
        self.tmp.cleanup()

    def collection(self, local_dir: Path) -> dict:
        return {
            "id": "collect_test",
            "source_type": "offline",
            "local_dir": str(local_dir),
            "status": "collecting_offline",
            "created_at": "2026-08-19T14:00:00+0800",
            "episodes": [
                {"id": "ep_000000", "name": "episode_000001",
                 "relative_path": "episode_000001.mp4", "status": "valid"},
                {"id": "ep_000001", "name": "episode_000002",
                 "relative_path": "episode_000002.mp4", "status": "invalid"},
                {"id": "ep_000002", "name": "episode_000003",
                 "relative_path": "episode_000003.mp4", "status": "unreviewed"},
            ],
        }

    def test_local_labels(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        collection = self.collection(local_dir)

        path = Path(app.write_local_labels(collection))

        self.assertEqual(path, local_dir / "labels.json")
        payload = json.loads(path.read_text("utf-8"))
        self.assertEqual(payload, {
            "session": {"valid_count": 1, "valid": [1], "invalid": [2]}
        })

    def test_refresh_restores_labels_by_relative_path(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        first = local_dir / "episode_000001.mp4"
        first.write_bytes(b"video-1")
        collection = self.collection(local_dir)
        collection["episodes"] = [collection["episodes"][0]]
        app.write_local_labels(collection)
        (local_dir / "episode_000002.mp4").write_bytes(b"video-2")
        collection["episodes"] = []

        app.refresh_collection_files(collection)

        statuses = {e["relative_path"]: e["status"] for e in collection["episodes"]}
        self.assertEqual(statuses["episode_000001.mp4"], "valid")
        self.assertEqual(statuses["episode_000002.mp4"], "unreviewed")
        self.assertEqual(collection["file_count"], 2)

    def test_refresh_still_reads_legacy_detailed_labels(self):
        local_dir = self.root / "legacy_session"
        local_dir.mkdir()
        (local_dir / "episode_000007.mp4").write_bytes(b"video")
        (local_dir / "labels.json").write_text(json.dumps({
            "version": 1,
            "valid": [{"relative_path": "episode_000007.mp4", "episode_number": 7}],
            "invalid": [], "unreviewed": [],
        }), "utf-8")
        collection = self.collection(local_dir)
        collection["episodes"] = []

        app.refresh_collection_files(collection)

        self.assertEqual(collection["episodes"][0]["status"], "valid")

    def test_episode_id_remains_stable_when_earlier_file_arrives(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        (local_dir / "episode_000002.mp4").write_bytes(b"video-2")
        collection = self.collection(local_dir)
        collection["episodes"] = []
        app.refresh_collection_files(collection)
        original_id = collection["episodes"][0]["id"]
        (local_dir / "episode_000001.mp4").write_bytes(b"video-1")

        app.refresh_collection_files(collection)

        episode = next(e for e in collection["episodes"]
                       if e["relative_path"] == "episode_000002.mp4")
        self.assertEqual(episode["id"], original_id)

    def test_start_offline_collection_has_no_remote_side_effects(self):
        local_dir = self.root / "live"
        skills_root = self.root / "skills"
        skills_root.mkdir()
        app.store.data["skills_config"] = {
            "skills_root": str(skills_root),
            "remote_base": "/remote",
            "host_id": "cluster_0",
        }
        skill = app.skill_backend.create_skill(
            {"id": "skill_1_walk", "title": "walk", "collect_root": str(self.root / "collections")},
            root=skills_root,
            host_id="cluster_0",
            remote_base="/remote",
        )
        with patch.object(app, "remote_mkdir", side_effect=AssertionError("network used")), \
             patch.object(app, "run_command", side_effect=AssertionError("network used")), \
             patch.object(app.subprocess, "Popen", side_effect=AssertionError("network used")), \
             patch.object(threading.Thread, "start"):
            response = app.app.test_client().post(
                "/api/collections/start",
                json={
                    "skill_id": skill["id"],
                    "local_dir": str(local_dir),
                    "auto_stack": False,
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()["collection"]
        self.assertEqual(payload["status"], "collecting_offline")
        self.assertEqual(payload["source_type"], "offline")
        self.assertEqual(payload["skill_id"], skill["id"])
        self.assertEqual(payload["host_id"], "cluster_0")
        self.assertTrue(local_dir.is_dir())

    def test_start_offline_collection_requires_registered_task(self):
        with patch.object(threading.Thread, "start"):
            response = app.app.test_client().post(
                "/api/collections/start",
                json={"local_dir": str(self.root / "live"), "auto_stack": False},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("技能", response.get_json()["message"])

    def test_worker_final_scan_writes_labels_and_waits_for_upload(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        (local_dir / "episode_000001.mp4").write_bytes(b"video")
        item = self.collection(local_dir)
        item["episodes"] = []
        app.store.data["collections"].append(item)
        stop = threading.Event()
        stop.set()

        with patch.object(app, "remote_mkdir", side_effect=AssertionError("network used")), \
             patch.object(app, "run_command", side_effect=AssertionError("network used")):
            app.offline_scan_worker(item["id"], stop)

        self.assertEqual(item["status"], "pending_upload")
        self.assertTrue((local_dir / "labels.json").is_file())
        self.assertEqual(len(item["episodes"]), 1)

    def test_mark_offline_episode_only_updates_local_labels(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        (local_dir / "episode_000001.mp4").write_bytes(b"video")
        item = self.collection(local_dir)
        item["episodes"] = [item["episodes"][0] | {"status": "unreviewed"}]
        app.store.data["collections"].append(item)

        with patch.object(app, "write_manifest", side_effect=AssertionError("remote used")):
            response = app.app.test_client().post(
                f"/api/collections/{item['id']}/mark",
                json={"episode_id": "ep_000000", "status": "valid"},
            )

        self.assertEqual(response.status_code, 200)
        labels = json.loads((local_dir / "labels.json").read_text("utf-8"))
        self.assertEqual(labels["session"], {
            "valid_count": 1, "valid": [1], "invalid": []
        })

    def test_manifest_preview_returns_root_labels_json(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        item = self.collection(local_dir)
        app.store.data["collections"].append(item)

        response = app.app.test_client().get(f"/api/collections/{item['id']}/manifest")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["filename"], "labels.json")
        self.assertEqual(payload["path"], str(local_dir / "labels.json"))
        self.assertEqual(payload["manifest"], {
            "session": {"valid_count": 1, "valid": [1], "invalid": [2]}
        })

    def upload_fixture(self):
        local_dir = self.root / "session"
        local_dir.mkdir()
        (local_dir / "episode_000001.mp4").write_bytes(b"video")
        item = self.collection(local_dir)
        item["status"] = "pending_upload"
        item["episodes"] = [item["episodes"][0]]
        app.write_local_labels(item)
        task = {
            "id": "task_1", "name": "skill_2_pick",
            "host_id": "cluster_0", "remote_dir": "/remote/skill_2",
        }
        item.update({"task_id": task["id"], "task_name": task["name"],
                     "host_id": task["host_id"]})
        app.store.data["tasks"].append(task)
        app.store.data["collections"].append(item)
        return item, task

    def test_upload_endpoint_rejects_path_outside_task(self):
        item, _ = self.upload_fixture()

        response = app.app.test_client().post(
            f"/api/collections/{item['id']}/upload",
            json={"task_id": "task_1", "host_id": "cluster_0",
                  "remote_dir": "/remote/other/session"},
        )

        self.assertEqual(response.status_code, 400)

    def test_upload_endpoint_binds_remote_target_and_starts_worker(self):
        item, _ = self.upload_fixture()
        with patch.object(threading.Thread, "start"):
            response = app.app.test_client().post(
                f"/api/collections/{item['id']}/upload",
                json={"task_id": "task_1", "host_id": "cluster_0",
                      "remote_dir": "/remote/skill_2/session"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(item["status"], "uploading")
        self.assertEqual(item["task_name"], "skill_2_pick")
        self.assertEqual(item["remote_dir"], "/remote/skill_2/session")

    def test_upload_rejects_changing_task_bound_before_collection(self):
        item, _ = self.upload_fixture()
        other = {"id": "task_2", "name": "skill_3", "host_id": "cluster_0",
                 "remote_dir": "/remote/skill_3"}
        app.store.data["tasks"].append(other)

        response = app.app.test_client().post(
            f"/api/collections/{item['id']}/upload",
            json={"task_id": other["id"], "host_id": "cluster_0",
                  "remote_dir": "/remote/skill_3/session"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("绑定任务", response.get_json()["message"])

    def test_remote_task_manifest_keeps_multiple_sessions(self):
        manifest = self.root / "skill_1.json"
        manifest.write_text(json.dumps({
            "session_a": {"valid_count": 1, "valid": [0], "invalid": [1]}
        }), "utf-8")
        second = {"valid_count": 2, "valid": [0, 2], "invalid": [1]}

        result = subprocess.run(
            [sys.executable, "-c", app.REMOTE_MANIFEST_MERGE,
             str(manifest), "session_b"],
            input=json.dumps(second), text=True, capture_output=True, check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        merged = json.loads(manifest.read_text("utf-8"))
        self.assertEqual(merged["session_a"]["valid"], [0])
        self.assertEqual(merged["session_b"], second)

    def test_deferred_upload_includes_labels_and_merges_remote_manifest(self):
        item, task = self.upload_fixture()
        item.update({"task_id": task["id"], "task_name": task["name"],
                     "host_id": "cluster_0", "remote_dir": "/remote/skill_2/session"})
        commands = []

        def capture(args, timeout=None):
            commands.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        merged = json.dumps({"session": {"valid_count": 1, "valid": [1], "invalid": []}})
        with patch.object(app, "remote_mkdir", return_value=(True, "")), \
             patch.object(app, "run_command", side_effect=capture), \
             patch.object(app.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0, merged, "")), \
             patch.object(app, "publish_collection_shared", return_value="shared_session"):
            app.deferred_upload_worker(item["id"])

        self.assertEqual(item["status"], "published")
        rsync_args = commands[0]
        self.assertEqual(rsync_args[0], "rsync")
        self.assertIn(str(Path(item["local_dir"])) + "/", rsync_args)
        self.assertNotIn("labels.json", " ".join(rsync_args))
        self.assertEqual(item["remote_manifest_file"], "/remote/skill_2/skill_2.json")

    def test_failed_upload_is_retryable(self):
        item, task = self.upload_fixture()
        item.update({"task_id": task["id"], "task_name": task["name"],
                     "host_id": "cluster_0", "remote_dir": "/remote/skill_2/session"})
        failure = subprocess.CompletedProcess([], 23, "", "network down")
        success = subprocess.CompletedProcess([], 0, "", "")
        merged = subprocess.CompletedProcess([], 0, json.dumps({"session": {}}), "")

        with patch.object(app, "remote_mkdir", return_value=(True, "")), \
             patch.object(app, "run_command", return_value=failure):
            app.deferred_upload_worker(item["id"])
        self.assertEqual(item["status"], "upload_error")

        with patch.object(app, "remote_mkdir", return_value=(True, "")), \
             patch.object(app, "run_command", return_value=success), \
             patch.object(app.subprocess, "run", return_value=merged), \
             patch.object(app, "publish_collection_shared", return_value="shared_session"):
            app.deferred_upload_worker(item["id"])
        self.assertEqual(item["status"], "published")

    def test_remote_manifest_rejects_labeled_episode_without_number(self):
        item, _ = self.upload_fixture()
        item["episodes"][0]["name"] = "episode_final"

        with self.assertRaisesRegex(ValueError, "无数字后缀"):
            app.remote_manifest_entry(item)


if __name__ == "__main__":
    unittest.main()
