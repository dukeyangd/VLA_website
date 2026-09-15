import json
import tempfile
import unittest
from pathlib import Path

import app
import infer_backend


ROOT = Path(__file__).resolve().parents[1]


class InferApiTests(unittest.TestCase):
    def setUp(self):
        self._state_dir = tempfile.TemporaryDirectory(prefix="studio_infer_test_")
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

    def test_catalog(self):
        with app.app.test_client() as client:
            resp = client.get("/api/infer/catalog")
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertGreaterEqual(len(data["catalog"]["skills"]), 1)
            self.assertEqual(data["catalog"]["defaults"]["rtc_inference_delay"], 6)

    def test_build_env(self):
        catalog = infer_backend.load_infer_catalog()
        job = {
            "student_ckpt": "/tmp/ckpt.pt",
            "ref_root": "/tmp/ds",
            "ep": 0,
            "out_dir": "/tmp/out",
            "stamp": "t",
            "params": {"prompt": "走", "use_rtc": True, "horizon": 32},
        }
        env = infer_backend.build_infer_env(job, catalog)
        self.assertEqual(env["HORIZON"], "32")
        self.assertEqual(env["USE_RTC"], "1")
        self.assertEqual(env["PHI0_RTC_INFERENCE_DELAY"], "6")
        self.assertEqual(env["PHI0_CL_PROMPT_OVERRIDE"], "走")

    def test_probe_paths(self):
        paths = infer_backend.episode_mp4_paths("/data/ds", 3)
        self.assertIn("episode_000003.mp4", paths["ego"])
        self.assertIn("left_wrist", paths["wrist"])

    def test_list_episode_indices(self):
        root = Path(tempfile.mkdtemp())
        ego = root / "videos/chunk-000/observation.images.ego_view"
        ego.mkdir(parents=True)
        (ego / "episode_000000.mp4").write_bytes(b"x")
        (ego / "episode_000002.mp4").write_bytes(b"x")
        self.assertEqual(infer_backend.list_episode_indices(str(root)), [0, 2])
        probe = infer_backend.probe_local_videos(str(root), 0)
        self.assertEqual(probe["episode_count"], 2)
        self.assertIn(0, probe["episodes"])

    def test_frontend_infer_page(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        js = (ROOT / "static" / "app.js").read_text("utf-8")
        self.assertIn('id="page-infer"', html)
        self.assertIn("Init Sim", html)
        self.assertIn("] 站立", html)
        self.assertIn("真机监视", html)
        self.assertIn("inferSimPreview", html)
        self.assertIn("切换技能", html)
        self.assertIn('id="inferSkillCards"', html)
        self.assertIn('id="inferSwitchPanel"', html)
        self.assertIn('id="inferActiveParams"', html)
        self.assertIn("更换视频", js)
        self.assertIn("function initInferPage()", js)
        self.assertIn("function renderInferSkillCards(", js)
        self.assertIn("function cycleInferReplayVideo(", js)
        self.assertIn("function applyInferSkillSwitch(", js)
        self.assertIn("function inferCmd(", js)
        self.assertIn("function startInferMonitor(", js)
        self.assertIn("function refreshInferPreview(", js)
        self.assertIn("btn-next", js)
        self.assertIn("/api/infer/jobs", js)
        self.assertNotIn("INFER · COMING SOON", html)
        self.assertNotIn("Horizon (chunk)", html)

    def test_interactive_phase_parse(self):
        info = infer_backend.parse_infer_phase("STUDIO phase=sim_ready Sim initialized")
        self.assertEqual(info["phase"], "sim_ready")
        info2 = infer_backend.studio_phase_from_status(
            '{"phase":"init_done","message":"Init Done"}',
            "noise",
        )
        self.assertEqual(info2["phase"], "init_done")


if __name__ == "__main__":
    unittest.main()
