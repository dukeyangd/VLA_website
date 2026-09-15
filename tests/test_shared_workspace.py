import copy
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


class SharedWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_store = app.store
        app.store = MemoryStore()

    def tearDown(self):
        app.store = self.old_store
        self.tmp.cleanup()

    def remote_call(self, db, action, payload):
        result = subprocess.run(
            [sys.executable, "-c", app.REMOTE_SHARED_DB, str(db), action],
            input=json.dumps(payload), text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def published_fixture(self):
        task = self.root / "skill_1_walk"
        session = task / "2026-08-19-10-00-00"
        session.mkdir(parents=True)
        video = session / "episode_000001.mp4"
        video.write_bytes(b"0123456789")
        payload = {
            "task": {"name": "skill_1_walk", "remote_dir": str(task),
                     "manifest_name": "skill_1.json", "created_at": "now", "updated_at": "now"},
            "session": {"id": "session_one", "name": session.name,
                        "remote_path": str(session), "source_type": "offline",
                        "source_hostname": "pc-one", "source_ip": "10.0.0.1",
                        "file_count": 1, "bytes_total": 10, "uploaded_at": "now"},
            "episodes": [{"id": "episode_one", "name": "episode_000001",
                          "relative_path": video.name,
                          "videos": [{"label": "ego", "relative_path": video.name}],
                          "status": "unreviewed"}],
        }
        return task, session, video, payload

    def test_shared_sqlite_publish_lock_mark_and_generated_files(self):
        task, session, video, payload = self.published_fixture()
        db = self.root / ".humanoid_data_studio" / "studio.db"
        self.remote_call(db, "publish", payload)

        first = self.remote_call(db, "lock", {
            "session_id": "session_one", "episode_id": "episode_one",
            "client_id": "client_a", "hostname": "pc-a", "ip": "10.0.0.1",
        })
        denied = self.remote_call(db, "lock", {
            "session_id": "session_one", "episode_id": "episode_one",
            "client_id": "client_b", "hostname": "pc-b", "ip": "10.0.0.2",
        })
        self.assertTrue(first["acquired"])
        self.assertFalse(denied["acquired"])
        self.assertEqual(denied["holder"]["client_id"], "client_a")

        marked = self.remote_call(db, "mark", {
            "session_id": "session_one", "episode_id": "episode_one",
            "client_id": "client_a", "hostname": "pc-a", "ip": "10.0.0.1",
            "status": "valid", "version": 0,
        })
        self.assertEqual(marked["version"], 1)
        labels = json.loads((session / "labels.json").read_text("utf-8"))
        manifest = json.loads((task / "skill_1.json").read_text("utf-8"))
        self.assertEqual(labels, {
            session.name: {"valid_count": 1, "valid": [1], "invalid": []}
        })
        self.assertEqual(manifest[session.name]["valid"], [1])

        listing = self.remote_call(db, "list", {})
        self.assertEqual(listing.get("detail"), "summary")
        self.assertEqual(listing["sessions"][0]["counts"]["valid"], 1)
        self.assertEqual(listing["sessions"][0]["episodes"], [])
        detail = self.remote_call(db, "get_session", {"session_id": "session_one"})
        episode = detail["session"]["episodes"][0]
        self.assertEqual(episode["status"], "valid")
        self.assertEqual(episode["version"], 1)

    def test_shared_mark_requires_lock_and_matching_version(self):
        _, _, _, payload = self.published_fixture()
        db = self.root / "studio.db"
        self.remote_call(db, "publish", payload)
        result = subprocess.run(
            [sys.executable, "-c", app.REMOTE_SHARED_DB, str(db), "mark"],
            input=json.dumps({"session_id": "session_one", "episode_id": "episode_one",
                              "client_id": "missing", "hostname": "pc", "ip": "1",
                              "status": "invalid", "version": 0}),
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lock", result.stderr.lower())

    def test_scan_existing_remote_session_reads_two_views_and_existing_labels(self):
        task = self.root / "skill_2_pick"
        session = task / "2026-08-19-12-00-00"
        head = session / "observation.images.head"
        wrist = session / "observation.images.left_wrist"
        head.mkdir(parents=True)
        wrist.mkdir(parents=True)
        (head / "episode_000001.mp4").write_bytes(b"head")
        (wrist / "episode_000001.mp4").write_bytes(b"wrist")
        (head / "episode_000002.mp4").write_bytes(b"second")
        (session / "labels.json").write_text(json.dumps({
            "version": 1,
            "valid": [{"relative_path": "observation.images.head/episode_000001.mp4",
                       "episode_number": 1}],
            "invalid": [], "unreviewed": [],
        }), "utf-8")
        (task / "skill_2.json").write_text(json.dumps({
            session.name: {"valid_count": 1, "valid": [1], "invalid": [2]}
        }), "utf-8")
        db = self.root / ".humanoid_data_studio" / "studio.db"
        self.remote_call(db, "register_task", {
            "name": "skill_2_pick", "remote_dir": str(task),
            "manifest_name": "skill_2.json", "created_at": "now", "updated_at": "now",
        })

        scanned = self.remote_call(db, "scan_session", {
            "task_name": "skill_2_pick", "remote_path": str(session),
        })
        self.assertEqual(len(scanned["episodes"]), 2)
        self.assertEqual(scanned["counts"], {"valid": 1, "invalid": 1, "unreviewed": 0})
        first = next(x for x in scanned["episodes"] if x["name"] == "episode_000001")
        self.assertEqual(len(first["videos"]), 2)
        self.assertEqual(first["videos"][0]["label"], "head")
        self.assertEqual(scanned["file_count"], 3)

        outside = subprocess.run(
            [sys.executable, "-c", app.REMOTE_SHARED_DB, str(db), "scan_session"],
            input=json.dumps({"task_name": "skill_2_pick", "remote_path": str(self.root)}),
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(outside.returncode, 0)
        self.assertIn("child", outside.stderr)

        conflict = subprocess.run(
            [sys.executable, "-c", app.REMOTE_SHARED_DB, str(db), "register_task"],
            input=json.dumps({"name": "skill_2_pick", "remote_dir": str(self.root / "other"),
                              "manifest_name": "skill_2.json", "created_at": "now",
                              "updated_at": "later"}),
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("different remote directory", conflict.stderr)

    def test_range_proxy_responses_and_unsatisfiable_range(self):
        info = {"ok": True, "path": "/remote/video.mp4", "size": 10,
                "mime": "video/mp4"}
        completed = subprocess.CompletedProcess([], 0, b"2345", b"")
        with patch.object(app, "shared_call", return_value=info), \
             patch.object(app, "shared_config", return_value=({}, {"target": "cluster_0"}, "/db")), \
             patch.object(app.subprocess, "run", return_value=completed):
            response = app.app.test_client().get(
                "/api/shared/media/session/episode/0", headers={"Range": "bytes=2-5"}
            )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"2345")
        self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")

        with patch.object(app, "shared_call", return_value=info), \
             patch.object(app, "shared_config", return_value=({}, {"target": "cluster_0"}, "/db")):
            response = app.app.test_client().get(
                "/api/shared/media/session/episode/0", headers={"Range": "bytes=99-100"}
            )
        self.assertEqual(response.status_code, 416)

    def test_publish_existing_remote_session_endpoint_scans_before_publish(self):
        scanned = {
            "ok": True,
            "task": {"name": "skill_4", "remote_dir": "/remote/skill_4",
                     "manifest_name": "skill_4.json", "created_at": "then",
                     "updated_at": "now"},
            "remote_path": "/remote/skill_4/session_a", "name": "session_a",
            "file_count": 2, "bytes_total": 20,
            "counts": {"valid": 1, "invalid": 0, "unreviewed": 0},
            "episodes": [{"id": "ep_one", "name": "episode_000001",
                          "relative_path": "episode_000001.mp4", "videos": [],
                          "status": "valid"}],
        }
        calls = []

        def fake_shared_call(action, payload, timeout=60):
            calls.append((action, payload))
            return scanned if action == "scan_session" else {
                "ok": True, "session_id": "session_shared"
            }

        with patch.object(app, "shared_call", side_effect=fake_shared_call), \
             patch.object(app, "local_identity", return_value={"hostname": "pc", "ip": "10.0.0.4"}):
            response = app.app.test_client().post(
                "/api/shared/remote-sessions/publish",
                json={"task_name": "skill_4", "remote_path": "/remote/skill_4/session_a"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([x[0] for x in calls], ["scan_session", "publish"])
        self.assertEqual(calls[1][1]["session"]["source_type"], "existing_remote")
        self.assertEqual(calls[1][1]["episodes"][0]["status"], "valid")

    def test_external_import_cannot_be_labeled_before_publish(self):
        item = {"id": "import_one", "source_type": "imported", "status": "importing",
                "episodes": [{"id": "ep", "status": "unreviewed"}]}
        app.store.data["collections"].append(item)
        response = app.app.test_client().post(
            "/api/collections/import_one/mark", json={"episode_id": "ep", "status": "valid"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("等待上传", response.get_json()["message"])


if __name__ == "__main__":
    unittest.main()
