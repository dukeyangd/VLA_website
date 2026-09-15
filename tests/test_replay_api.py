import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class ReplayApiTests(unittest.TestCase):
    def setUp(self):
        self._state_dir = tempfile.TemporaryDirectory(prefix="studio_replay_test_")
        self._prev_state_file = app.STATE_FILE
        self._prev_state_dir = app.STATE_DIR
        app.STATE_DIR = Path(self._state_dir.name)
        app.STATE_FILE = app.STATE_DIR / "studio.json"
        app.store.data = app.default_state()
        app.store.save()

    def tearDown(self):
        app.STATE_FILE = self._prev_state_file
        app.STATE_DIR = self._prev_state_dir
        self._state_dir.cleanup()

    def test_parse_episode_selection(self):
        self.assertEqual(app.parse_episode_selection("60-70", 100), list(range(60, 71)))
        self.assertEqual(app.parse_episode_selection("62 63 64", 100), [62, 63, 64])
        with self.assertRaises(ValueError):
            app.parse_episode_selection("99-101", 100)

    def test_translate_replay_fail(self):
        text = app.translate_replay_fail({"episode": 63, "reasons": ["fall", "sim_twitch"], "fall_step": 120})
        self.assertIn("Episode 63", text)
        self.assertIn("摔倒", text)
        self.assertIn("抖动", text)

    def test_viser_tunnel_cmd_keeps_local_forward(self):
        cmd = app._viser_tunnel_cmd("cluster_0", 8081)
        self.assertIn("-L", cmd)
        self.assertIn("8081:127.0.0.1:8081", cmd)
        self.assertIn("-N", cmd)
        self.assertNotIn("ClearAllForwardings=yes", cmd)
        st = app.viser_tunnel_status()
        self.assertFalse(st["alive"])
        with app.app.test_client() as client:
            resp = client.get("/api/replay/viser-tunnel")
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertIn("tunnel", data)

    def test_replay_config_endpoint(self):
        with app.app.test_client() as client:
            resp = client.get("/api/replay/config")
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertIn("presets", data)
            self.assertTrue(data.get("local_qa_exists") or data.get("qa_root_default"))

    def test_local_dataset_probe(self):
        path = "/home/neotix/noetix/dataprocess/skill_2_pico_new826_unified"
        if not __import__("pathlib").Path(path).is_dir():
            self.skipTest("local dataprocess dataset not present")
        with app.app.test_client() as client:
            resp = client.post("/api/replay/datasets/probe", json={"dataset_path": path})
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertGreater(data["total_episodes"], 0)
            self.assertEqual(data["execution_mode"], "local")
            self.assertTrue(len(data.get("experiments") or []) >= 0)

    def test_issues_roundtrip(self):
        path = "/mnt/data2/wpy/test_dataset"
        with app.app.test_client() as client:
            resp = client.post("/api/replay/issues", json={
                "dataset_path": path,
                "marked_delete": [3, 5],
                "issue_updates": {"3": {"note": "仿真抖动", "status": "fail", "summary_zh": "Episode 3：抖动"}},
            })
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertEqual(data["issues"]["marked_delete"], [3, 5])
            self.assertEqual(data["issues"]["issues"]["3"]["note"], "仿真抖动")

    def test_merge_summary_union_keeps_historical_fails(self):
        path = "/mnt/data2/wpy/union_dataset"
        app.merge_replay_summary(path, {"ok": [7, 8], "fail": [{"episode": 7, "reasons": ["fall"]}]})
        app.merge_replay_summary(path, {"ok": [7, 8], "fail": [{"episode": 8, "reasons": ["sim_twitch"]}]})
        issues = app.issues_for_dataset(path)
        self.assertIn("7", issues["issues"])
        self.assertIn("8", issues["issues"])

    def test_reset_session_clears_transient_state(self):
        path = "/mnt/data2/wpy/reset_dataset"
        with app.app.test_client() as client:
            client.post("/api/replay/issues", json={
                "dataset_path": path,
                "marked_delete": [1, 2],
                "issue_updates": {"1": {"status": "fail", "summary_zh": "Episode 1：fail"}},
            })
            app.issues_for_dataset(path)["prune_status"] = "completed"
            app.store.save()
            resp = client.post("/api/replay/issues/reset-session", json={"dataset_path": path})
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertEqual(data["issues"]["marked_delete"], [])
            self.assertEqual(data["issues"]["prune_status"], "pending")
            self.assertIn("1", data["issues"]["issues"])

    def test_clear_issues_log(self):
        path = "/mnt/data2/wpy/clear_dataset"
        with app.app.test_client() as client:
            client.post("/api/replay/issues", json={
                "dataset_path": path,
                "issue_updates": {"2": {"status": "fail", "summary_zh": "Episode 2：fail"}},
            })
            resp = client.post("/api/replay/issues/clear", json={"dataset_path": path})
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertEqual(data["issues"]["issues"], {})
            self.assertEqual(data["issues"]["marked_delete"], [])

    def test_parse_labels_valid_only(self):
        parsed = app.parse_labels_valid_invalid({
            "session_a": {"valid_count": 2, "valid": [0, 2], "invalid": [1]},
            "session_b": {"valid_count": 1, "valid": [3], "invalid": []},
        })
        self.assertEqual(parsed["valid"], [0, 2, 3])
        self.assertEqual(parsed["invalid"], [1])
        with app.app.test_client() as client:
            resp = client.post("/api/replay/parse-labels", json={
                "labels": {"valid": [1, 5, 9], "invalid": [2, 3]},
                "total_episodes": 10,
            })
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertEqual(data["valid"], [1, 5, 9])

    def test_export_valid_allowlist_from_issues(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds"
            meta = root / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text(json.dumps({
                "total_episodes": 5, "fps": 30, "robot_type": "g1",
            }), encoding="utf-8")
            # minimal episodes parquet not required if probe falls back — use local probe mock
            path = str(root)
            app.merge_replay_summary(path, {"fail": [{"episode": 1, "reasons": ["fall"]}]})
            app.merge_replay_summary(path, {"fail": [{"episode": 3, "reasons": ["sim_twitch"]}]})
            with app.app.test_client() as client:
                with patch("app.probe_dataset", return_value={
                    "path": path, "name": "ds", "total_episodes": 5, "total_frames": 0, "fps": 30,
                }), patch("app.use_local_execution", return_value=True):
                    resp = client.post("/api/replay/export-valid-allowlist", json={
                        "dataset_path": path, "source": "issues",
                    })
            data = json.loads(resp.data)
            self.assertTrue(data["ok"], data)
            self.assertEqual(data["valid"], [0, 2, 4])
            self.assertEqual(data["invalid"], [1, 3])
            allow = json.loads((meta / "vision_episode_allowlist.json").read_text("utf-8"))
            self.assertEqual(allow["episode_index"], [0, 2, 4])
            qa = json.loads((meta / "sonic_qa_valid_invalid.json").read_text("utf-8"))
            self.assertEqual(qa["valid_count"], 3)


if __name__ == "__main__":
    unittest.main()
