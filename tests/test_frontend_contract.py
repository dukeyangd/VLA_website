import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_offline_collection_and_upload_controls_exist(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        js = (ROOT / "static" / "app.js").read_text("utf-8")

        self.assertIn('id="localDir"', html)
        self.assertIn('id="collectionTask"', html)
        self.assertIn("开始离线数采", html)
        self.assertIn('id="uploadModal"', html)
        self.assertIn('id="uploadTask"', html)
        self.assertIn('id="uploadSkillGrid"', html)
        self.assertIn("function uploadCollection()", js)
        self.assertIn("function retryUpload(", js)
        self.assertIn("重新上传", js)
        self.assertIn("selectUploadSkill", js)
        self.assertIn("混合训练", html)
        self.assertIn("enterTrainMixMode", js)
        self.assertNotIn("新增技能训练", html)
        self.assertNotIn("开始数采与实时上传", html)
        self.assertNotIn("数采实时同步已启动", js)

    def test_offline_start_binds_cached_task_without_remote_destination(self):
        js = (ROOT / "static" / "app.js").read_text("utf-8")
        start = js[js.index("async function startCollection"):]
        start = start[:start.index("async function", 20)]

        self.assertIn("local_dir", start)
        self.assertIn("skill_id", start)
        self.assertNotIn("host_id", start)
        self.assertNotIn("remote_dir", start)

    def test_shared_workspace_and_lock_controls_exist(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        js = (ROOT / "static" / "app.js").read_text("utf-8")

        self.assertIn('id="sharedSessionList"', html)
        self.assertIn('id="sharedModal"', html)
        self.assertIn('id="lockNotice"', html)
        self.assertIn("/api/shared/locks/acquire", js)
        self.assertIn("/api/shared/media/", js)
        self.assertIn("/api/shared/mark", js)
        self.assertIn('id="remotePublishModal"', html)
        self.assertIn("/api/shared/remote-sessions/scan", js)
        self.assertIn("/api/shared/remote-sessions/publish", js)
        self.assertIn("leavingReview", js)

    def test_replay_qa_controls_exist(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        js = (ROOT / "static" / "app.js").read_text("utf-8")

        self.assertIn('data-page="skills"', html)
        self.assertIn('id="page-skills"', html)
        self.assertIn('id="mgmtSkillGrid"', html)
        self.assertIn('id="mgmtDatasetPath"', html)
        self.assertIn("createManagedSkill", js)
        self.assertIn("deleteManagedSkill", js)
        self.assertIn("addManagedDataset", js)
        self.assertIn("removeManagedDataset", js)
        self.assertIn('data-page="replay"', html)
        self.assertIn('id="page-replay"', html)
        self.assertIn('id="viserFrame"', html)
        self.assertIn('id="issueBoard"', html)
        self.assertIn('id="replayDatasetPath"', html)
        self.assertIn('id="replaySkillGrid"', html)
        self.assertIn('id="replayAttachSkill"', html)
        self.assertIn('id="replayAttachPath"', html)
        self.assertIn('data-page="train"', html)
        self.assertIn('data-page="infer"', html)
        self.assertIn("selectReplaySkill", js)
        self.assertIn("attachDatasetToReplaySkill", js)
        self.assertIn("/api/skills/", js)
        self.assertIn("/api/replay/datasets/probe", js)
        self.assertIn("/api/replay/sessions", js)
        self.assertIn("/api/replay/prune", js)
        self.assertIn("function startReplay()", js)
        self.assertIn("function confirmPrune()", js)
        prune = (ROOT / "sonic_qa" / "prune_lerobot_v3_episodes.py").read_text("utf-8")
        self.assertTrue(prune.startswith("#!/usr/bin/env python3"))
        self.assertIn("v2.1", prune)
        self.assertIn("detect_format", prune)
        self.assertIn("datasetFormatLabel", js)


if __name__ == "__main__":
    unittest.main()
