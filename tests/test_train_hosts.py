"""Unit tests for multi-host train path helpers."""
from __future__ import annotations

import unittest

import train_backend


class TrainHostsTest(unittest.TestCase):
    def test_builtin_hosts_merged(self):
        merged = train_backend.merge_builtin_train_hosts(
            [{"id": "cluster_0", "name": "cluster_0", "target": "cluster_0"}]
        )
        ids = [h["id"] for h in merged]
        self.assertEqual(ids[:4], ["cluster_0", "h20-0", "h20-1", "cluster_2"])
        h20 = next(h for h in merged if h["id"] == "h20-0")
        self.assertEqual(h20["model_zoo"], train_backend.DEFAULT_TRAIN_OUT_BASE)
        self.assertEqual(h20["train_data"], train_backend.DEFAULT_TRAIN_PACK_BASE)
        self.assertFalse(h20["train_ready"])

    def test_preserve_train_ready(self):
        merged = train_backend.merge_builtin_train_hosts(
            [{"id": "h20-1", "train_ready": True, "train_ready_reason": ""}]
        )
        h = next(x for x in merged if x["id"] == "h20-1")
        self.assertTrue(h["train_ready"])

    def test_offbox_detection(self):
        self.assertTrue(train_backend.is_offbox_train_host("h20-0"))
        self.assertTrue(train_backend.is_offbox_train_host("cluster_2"))
        self.assertFalse(train_backend.is_offbox_train_host("cluster_0"))
        self.assertFalse(train_backend.is_offbox_train_host("local"))

    def test_resolve_train_phi0_py_prefers_newton(self):
        py = train_backend.resolve_train_phi0_py()
        self.assertIn("newton", py.lower())
        self.assertNotIn("Phi-0-wpy", py)

    def test_parse_metrics_jsonl_bytes_offset(self):
        lines = [
            '{"step":1,"loss":1.0}\n',
            '{"step":2,"loss":0.5}\n',
            '{"step":3,"loss":0.2}\n',
        ]
        text = "".join(lines)
        off = len(lines[0].encode("utf-8"))
        rows = train_backend.parse_metrics_jsonl_bytes(text, byte_offset=off)
        self.assertEqual([r["step"] for r in rows], [2, 3])
        self.assertEqual(train_backend.parse_metrics_jsonl_bytes(text, byte_offset=10_000), [])

    def test_collect_stage_and_pullback_paths(self):
        job = {
            "ref_root": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data/skill_x/studio_tmp_unified_link",
            "out_dir": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/skill_x_20260101",
            "episode_allowlist": "/tmp/allow.json",
            "pack": {
                "pack_root": "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data/skill_x_20260101",
                "manifest_path": "/tmp/man.json",
                "stage_sources": {
                    "sess_a": "/mnt/data2/wpy/workspace/830demo/skill1/sess_a",
                },
                "session_paths": ["/mnt/data2/wpy/workspace/830demo/skill1/sess_a"],
            },
        }
        staged = train_backend.collect_train_stage_paths(job)
        self.assertIn("/tmp/man.json", staged)
        self.assertIn("/mnt/data2/wpy/workspace/830demo/skill1/sess_a", staged)
        self.assertIn("/tmp/allow.json", staged)
        pull = train_backend.pullback_paths_for_job(job)
        self.assertEqual(
            pull,
            [
                "/mnt/data2/wpy/workspace/Phi_0_model_zoo/skill_x_20260101",
                "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data/skill_x_20260101",
            ],
        )

    def test_archive_train_pack_to_efs(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            local_base = Path(td) / "local_packs"
            archive_base = Path(td) / "efs_packs"
            local_base.mkdir()
            archive_base.mkdir()
            src = local_base / "skillX_t"
            (src / "meta").mkdir(parents=True)
            (src / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            (src / "studio_tmp_unified_link").symlink_to(src / "meta")
            # Point DEFAULT temporarily via archive_base arg; safety check uses DEFAULT.
            # Put src under real DEFAULT by monkeypatching constant.
            old = train_backend.DEFAULT_TRAIN_PACK_BASE
            try:
                train_backend.DEFAULT_TRAIN_PACK_BASE = str(local_base)
                result = train_backend.archive_train_pack_to_efs(
                    {"pack_root": str(src)},
                    archive_base=str(archive_base),
                    delete_local=True,
                )
            finally:
                train_backend.DEFAULT_TRAIN_PACK_BASE = old
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["archived"], result)
            self.assertTrue(result["deleted_local"], result)
            self.assertTrue((archive_base / "skillX_t" / "meta" / "stats.json").is_file())
            self.assertFalse(src.exists())

    def test_remote_pack_symlink_bash(self):
        bash = train_backend.remote_pack_symlink_bash(
            {
                "raw_root": "/mnt/data2/x/raw_stage",
                "stage_sources": {"s1": "/mnt/data2/wpy/workspace/830demo/s1"},
                "pack_root": "/mnt/data2/x",
                "nvme_dir": "/mnt/data2/x/studio_tmp_unified",
                "ws_link": "/mnt/data2/x/studio_tmp_unified_link",
            }
        )
        self.assertIn("ln -s", bash)
        self.assertIn("raw_stage", bash)
        self.assertIn("studio_tmp_unified_link", bash)


if __name__ == "__main__":
    unittest.main()
