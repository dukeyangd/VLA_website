import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import train_backend


ROOT = Path(__file__).resolve().parents[1]


class TrainApiTests(unittest.TestCase):
    def setUp(self):
        self._state_dir = tempfile.TemporaryDirectory(prefix="studio_train_test_")
        self._prev_state_file = app.STATE_FILE
        self._prev_state_dir = app.STATE_DIR
        app.STATE_DIR = Path(self._state_dir.name)
        app.STATE_FILE = app.STATE_DIR / "studio.json"
        app.store.data = app.default_state()
        skills_root = app.STATE_DIR / "test_skills"
        skills_root.mkdir(parents=True, exist_ok=True)
        app.store.data["skills_config"] = {
            "skills_root": str(skills_root),
            "remote_base": "/tmp/studio_skills_remote",
            "host_id": "cluster_0",
        }
        app.store.save()

    def tearDown(self):
        app.STATE_FILE = self._prev_state_file
        app.STATE_DIR = self._prev_state_dir
        self._state_dir.cleanup()

    def test_catalog_endpoint(self):
        with app.app.test_client() as client:
            resp = client.get("/api/train/catalog")
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertEqual(data["catalog"].get("source_mode"), "recipes+skills")
            # Empty test skills root → builtin recipes only until a skill is created.
            self.assertTrue(any(t.get("id") == "skill1_walk" for t in data["catalog"]["tasks"]))
            created = json.loads(client.post("/api/skills", json={
                "title": "走向黑箱", "id": "walk_blackbox", "badge": "WALK", "prompt": "走",
            }).data)
            self.assertTrue(created["ok"], created)
            data2 = json.loads(client.get("/api/train/catalog").data)
            self.assertIn("walk_blackbox", [t["id"] for t in data2["catalog"]["tasks"]])
            walk = next(t for t in data2["catalog"]["tasks"] if t["id"] == "walk_blackbox")
            self.assertEqual(walk["defaults"]["epochs"], 4)
            self.assertEqual(walk["defaults"]["ckpt_every"], 10000)
            self.assertEqual(walk.get("train_profile"), "vision_teleop_handcmd_rtc")
            self.assertEqual(walk.get("script"), "tools/train/run_online_vlm_mix_distill.sh")
            self.assertEqual(walk.get("defaults").get("extra_steps", 0), 0)
            self.assertFalse(data2["catalog"].get("default_init_student_ckpt"))
            self.assertFalse(walk.get("default_student_ckpt"))

    def test_probe_local_dataset(self):
        path = "/home/neotix/noetix/dataprocess/skill_walk_to_black_box_new_unified"
        if not Path(path).is_dir():
            self.skipTest("local dataset missing")
        with app.app.test_client() as client:
            resp = client.post("/api/train/datasets/probe", json={"dataset_path": path})
            data = json.loads(resp.data)
            self.assertTrue(data["ok"])
            self.assertTrue(data["exists"])
            self.assertTrue(data.get("prompt"))

    def test_parse_metrics_jsonl(self):
        text = "\n".join([
            json.dumps({"chunk": 1, "loss": 1.2, "loss_z": 0.5}),
            json.dumps({"chunk": 2, "loss": 1.0, "loss_z": 0.4, "grad_l2": 0.9}),
            "not-json",
        ])
        rows = train_backend.parse_metrics_jsonl(text)
        self.assertEqual(len(rows), 2)
        live = train_backend.summarize_metrics(rows, {"steps_per_epoch": 100, "epochs": 10})
        self.assertEqual(live["steps_done"], 2)
        self.assertAlmostEqual(live["epoch_frac"], 0.02)
        self.assertEqual(live["eta_steps"], 998)
        self.assertIsNotNone(live.get("loss_smooth"))
        smoothed = train_backend.smooth_metric_series(rows)
        self.assertIn("loss_smooth", smoothed[-1])

    def test_metrics_full_history_downsample(self):
        # Previously UI kept only the last N rows; chart must span step 0 → end.
        lines = [json.dumps({"chunk": i, "loss": 1.0 / (i + 1)}) for i in range(5000)]
        rows = train_backend.parse_metrics_jsonl("\n".join(lines))
        self.assertEqual(len(rows), 5000)
        self.assertEqual(rows[0]["step"], 0)
        self.assertEqual(rows[-1]["step"], 4999)
        chart = train_backend.downsample_metric_rows(rows, 800)
        self.assertLessEqual(len(chart), 800)
        self.assertEqual(chart[0]["step"], 0)
        self.assertEqual(chart[-1]["step"], 4999)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "distill_metrics.jsonl"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            from_file = train_backend.parse_metrics_jsonl_file(p)
            self.assertEqual(len(from_file), 5000)
            self.assertEqual(from_file[0]["step"], 0)
            # byte_offset skips prior-run bytes (reused out_dir).
            mid = p.read_bytes().find(b"\n", 1000) + 1
            after = train_backend.parse_metrics_jsonl_file(p, byte_offset=mid)
            self.assertLess(len(after), 5000)
            self.assertGreater(after[0]["step"], 0)

    def test_metrics_file_size_helper(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "distill_metrics.jsonl"
            self.assertEqual(train_backend.metrics_file_size(p), 0)
            p.write_text('{"chunk":1,"loss":1}\n', encoding="utf-8")
            self.assertEqual(train_backend.metrics_file_size(p), p.stat().st_size)

    def test_parse_distill_log_metrics(self):
        text = "\n".join([
            "[distill]   101/15290   0.7% │─│   3.2 step/s (w50=3.8)  ETA 1.1h  loss=0.6853  z=0.427  z_phys=0.120  hand=0.258  fall_cum=0",
            "[distill]   101/15290   0.7% │─│   3.2 step/s (w50=3.8)  ETA 1.1h  loss=0.6853  z=0.427  z_phys=0.120  hand=0.258  fall_cum=0",
            "[distill]  1001/15290   6.5% │━│   3.6 step/s (w50=3.5)  ETA 1.0h  loss=0.1907  z=0.143  z_phys=0.040  hand=0.047  fall_cum=0",
        ])
        rows, meta = train_backend.parse_distill_log_metrics(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["step"], 1001)
        self.assertAlmostEqual(rows[-1]["loss"], 0.1907)
        self.assertAlmostEqual(rows[-1]["loss_z"], 0.143)
        self.assertAlmostEqual(rows[-1]["loss_z_phys"], 0.040)
        self.assertAlmostEqual(rows[-1]["loss_hand"], 0.047)
        self.assertAlmostEqual(rows[-1]["steps_per_sec"], 3.6)
        self.assertEqual(meta.get("max_steps"), 15290)
        live = train_backend.summarize_metrics(
            rows, {**meta, "epochs": 10, "steps_per_epoch": 1529},
            params={"epochs": 10},
        )
        self.assertEqual(live["steps_done"], 1001)
        self.assertIsNotNone(live.get("loss_smooth"))
        self.assertAlmostEqual(live["steps_per_sec"], 3.6)
        self.assertAlmostEqual(live["loss_hand"], 0.047)

    def test_list_student_ckpts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "phi0_student_last.pt").write_bytes(b"x")
            (root / "phi0_student_step000100.pt").write_bytes(b"y")
            (root / "skill4_20260914_153627.pt").write_bytes(b"z")
            (root / "skill4_20260914_153627_optim.pt").write_bytes(b"o")
            (root / "info.json").write_text(json.dumps({
                "skill": "skill4",
                "data": {"paths": ["/data/s1"], "episode_count": 12, "session_count": 2},
                "checkpoints": [
                    {"name": "skill4_20260914_153627.pt", "loss": 0.041, "steps": 100},
                ],
            }), encoding="utf-8")
            found = train_backend.list_student_ckpts(str(root))
            names = [c["name"] for c in found]
            self.assertIn("phi0_student_last.pt", names)
            self.assertIn("phi0_student_step000100.pt", names)
            self.assertIn("skill4_20260914_153627.pt", names)
            self.assertNotIn("skill4_20260914_153627_optim.pt", names)
            named = next(c for c in found if c["name"] == "skill4_20260914_153627.pt")
            self.assertEqual(named["loss"], 0.041)
            self.assertEqual(named["episode_count"], 12)

    def test_train_ckpt_data_meta_and_env(self):
        catalog = train_backend.load_train_catalog()
        job = {
            "skill_id": "skill4",
            "task_id": "skill4",
            "ref_root": "/tmp/ds",
            "out_dir": "/tmp/out",
            "log_file": "/tmp/out.log",
            "stamp": "t",
            "train_profile": "vision_teleop",
            "params": {
                "ngpu": 8, "num_envs": 32, "horizon": 32,
                "epochs": 2, "ckpt_every": 2000, "ckpt_every_epoch": 0,
                "dataset_paths": ["/data/skill4/s1", "/data/skill4/s2"],
                "valid_count": 40,
            },
            "pack": {
                "session_count": 2,
                "valid_count": 40,
                "stage_sources": {"s1": "/data/skill4/s1", "s2": "/data/skill4/s2"},
            },
        }
        meta = train_backend.train_ckpt_data_meta(job)
        self.assertEqual(meta["episode_count"], 40)
        self.assertEqual(meta["session_count"], 2)
        self.assertEqual(meta["paths"], ["/data/skill4/s1", "/data/skill4/s2"])
        env = train_backend.build_train_env(job, catalog)
        self.assertEqual(env["PHI0_CKPT_SKILL"], "skill4")
        data = json.loads(env["PHI0_TRAIN_DATA_META"])
        self.assertEqual(data["episode_count"], 40)

    def test_build_train_env_epochs(self):
        catalog = train_backend.load_train_catalog()
        job = {
            "ref_root": "/tmp/ds",
            "out_dir": "/tmp/out",
            "log_file": "/tmp/out.log",
            "stamp": "t",
            "train_profile": "vision_teleop",
            "params": {
                "ngpu": 8, "num_envs": 32, "horizon": 32,
                "epochs": 10, "ckpt_every": 2000, "ckpt_every_epoch": 0,
                "lr": "1e-4", "prompt": "走",
            },
        }
        env = train_backend.build_train_env(job, catalog)
        self.assertEqual(env["EFFECTIVE_BATCH"], "256")
        self.assertEqual(env["EPOCHS"], "10")
        self.assertNotIn("EXTRA_STEPS", env)
        self.assertEqual(env["CKPT_EVERY"], "2000")
        self.assertEqual(env["PHI0_CKPT_EVERY_EPOCH"], "0")
        self.assertEqual(env["PHI0_CKPT_STEP_KEEP"], "3")
        self.assertEqual(env["VISION_ONLY"], "1")
        self.assertEqual(env["PROMPT"], "走")

    def test_build_train_env_epoch_ckpt_default(self):
        catalog = train_backend.load_train_catalog()
        job = {
            "ref_root": "/tmp/ds",
            "out_dir": "/tmp/out",
            "log_file": "/tmp/out.log",
            "stamp": "t",
            "train_profile": "vision_teleop",
            "params": {
                "ngpu": 8, "num_envs": 32, "horizon": 32,
                "epochs": 3, "lr": "1e-4",
            },
        }
        env = train_backend.build_train_env(job, catalog)
        self.assertEqual(env["PHI0_CKPT_EVERY_EPOCH"], "1")
        self.assertEqual(env["CKPT_EVERY"], "0")

    def test_train_profiles_and_launcher(self):
        catalog = train_backend.load_train_catalog()
        self.assertEqual(
            train_backend.launcher_script(catalog),
            "tools/train/run_online_vlm_mix_distill.sh",
        )
        self.assertIn("vision_teleop_handcmd_rtc", train_backend.TRAIN_PROFILES)
        self.assertIn("mix_vision_isaac", train_backend.TRAIN_PROFILES)
        walk = next(t for t in catalog["tasks"] if t["id"] == "skill1_walk")
        self.assertEqual(walk["train_profile"], "vision_teleop_handcmd_rtc")
        self.assertEqual(walk["script"], "tools/train/run_online_vlm_mix_distill.sh")
        job = {
            "ref_root": "/tmp/ds",
            "out_dir": "/tmp/out",
            "log_file": "/tmp/out.log",
            "stamp": "t",
            "train_profile": "mix_vision_isaac",
            "params": {"ngpu": 8, "num_envs": 32, "horizon": 32, "epochs": 2, "ckpt_every": 1000},
        }
        env = train_backend.build_train_env(job, catalog)
        self.assertEqual(env["PHI0_P_VISION"], "0.9")
        self.assertEqual(env["VISION_ONLY"], "0")
        self.assertEqual(env["PHI0_ALLOW_ZERO_HAND"], "1")
        with self.assertRaises(ValueError):
            train_backend.validate_train_ref(
                "/mnt/data2/wpy/workspace/local_nvme/datasets/830/830mix_skill1234_demo5_unified",
                train_profile="vision_teleop",
                require_local_files=False,
            )
        with self.assertRaises(ValueError):
            train_backend.validate_train_ref(
                "/mnt/data2/wpy/workspace/830demo/skill6/2026-09-09-16-31-40",
                train_profile="vision_teleop",
                require_local_files=False,
            )

    def test_build_train_env_no_student_ckpt(self):
        catalog = train_backend.load_train_catalog()
        job = {
            "ref_root": "/tmp/ds",
            "out_dir": "/tmp/out",
            "log_file": "/tmp/out.log",
            "stamp": "t",
            "params": {
                "ngpu": 8, "num_envs": 32, "horizon": 32,
                "epochs": 10, "extra_steps": 0, "ckpt_every": 2000,
                "lr": "1e-4", "prompt": "走",
            },
        }
        env = train_backend.build_train_env(job, catalog)
        self.assertEqual(env["EPOCHS"], "10")
        self.assertNotIn("EXTRA_STEPS", env)
        self.assertNotIn("STUDENT_CKPT", env)

    def test_skill_card_appears_in_train_catalog(self):
        with app.app.test_client() as client:
            resp = client.post("/api/skills", json={
                "title": "测试开门",
                "id": "skill_open_test",
                "badge": "OPEN",
                "prompt": "打开门",
            })
            data = json.loads(resp.data)
            self.assertTrue(data["ok"], data)
            sid = data["skill"]["id"]
            self.assertEqual(sid, "skill_open_test")
            cat = client.get("/api/train/catalog")
            cat_data = json.loads(cat.data)
            ids = [t["id"] for t in cat_data["catalog"]["tasks"]]
            self.assertIn(sid, ids)
            mix = json.loads(client.post("/api/train/mix", json={"skill_ids": [sid]}).data)
            self.assertFalse(mix.get("ok"))
            # create second skill then mix
            client.post("/api/skills", json={"title": "走", "id": "skill_walk_test", "badge": "WALK"})
            mix2 = json.loads(client.post("/api/train/mix", json={"skill_ids": [sid, "skill_walk_test"]}).data)
            self.assertTrue(mix2["ok"], mix2)
            self.assertTrue(mix2["task"].get("mix"))
            self.assertEqual(mix2["task"]["badge"], "MIX")
            self.assertIn("mix_slots", mix2["task"])
            self.assertEqual(len(mix2["task"]["mix_slots"]), 2)
            self.assertEqual(mix2["task"]["train_profile"], "mix_vision_isaac")

    def test_catalog_has_no_resume_task(self):
        with app.app.test_client() as client:
            data = json.loads(client.get("/api/train/catalog").data)
            ids = [t["id"] for t in data["catalog"]["tasks"]]
            self.assertNotIn("resume_extra", ids)
            self.assertFalse(any(t.get("requires_ckpt") for t in data["catalog"]["tasks"]))

    def test_fresh_train_has_no_student_ckpt(self):
        with patch("app.threading.Thread") as th:
            th.return_value.start = lambda: None
            with app.app.test_client() as client:
                client.post("/api/skills", json={"title": "走", "id": "walk_for_job", "badge": "WALK"})
                resp = client.post("/api/train/jobs", json={
                    "task_id": "walk_for_job",
                    "dataset_path": "/home/neotix/noetix/dataprocess/skill_walk_to_black_box_new_unified",
                    "execution_mode": "remote",
                    "params": {"epochs": 2, "num_envs": 32, "ngpu": 8},
                })
                data = json.loads(resp.data)
                self.assertTrue(data["ok"], data)
                job = data["job"]
                self.assertFalse(job.get("student_ckpt"))
                self.assertEqual(job["params"]["extra_steps"], 0)
                self.assertFalse(job["params"].get("init_from_base"))
                self.assertEqual(job["params"]["epochs"], 2)
                env = train_backend.build_train_env(job, train_backend.load_train_catalog())
                self.assertNotIn("STUDENT_CKPT", env)
                self.assertNotIn("PHI0_RESUME_STEP0", env)
                self.assertNotIn("EXTRA_STEPS", env)

    def test_add_existing_dataset_to_skill(self):
        path = "/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/skill_2_pico_new826_unified"
        with patch("app.probe_dataset") as probe:
            probe.return_value = {"name": "skill_2_pico_new826_unified", "total_episodes": 42}
            with app.app.test_client() as client:
                created = json.loads(client.post("/api/skills", json={
                    "title": "抓玩具", "id": "skill_pick_toy", "badge": "PICK",
                }).data)
                self.assertTrue(created["ok"], created)
                resp = client.post("/api/skills/skill_pick_toy/datasets", json={
                    "path": path,
                    "label": "pico826",
                    "host_id": "cluster_0",
                    "probe": True,
                })
                data = json.loads(resp.data)
                self.assertTrue(data["ok"], data)
                paths = [d.get("path") for d in data.get("datasets") or []]
                self.assertIn(path, paths)
                listed = json.loads(client.get("/api/skills/skill_pick_toy/datasets").data)
                self.assertTrue(listed["ok"])
                self.assertTrue(any(d.get("path") == path for d in listed["datasets"]))
                probe.assert_called()

                removed = json.loads(client.delete("/api/skills/skill_pick_toy/datasets", json={"path": path}).data)
                self.assertTrue(removed["ok"], removed)
                self.assertFalse(any(d.get("path") == path for d in removed.get("datasets") or []))

                deleted = json.loads(client.delete("/api/skills/skill_pick_toy").data)
                self.assertTrue(deleted["ok"], deleted)
                missing = client.get("/api/skills/skill_pick_toy")
                self.assertEqual(missing.status_code, 404)

class TrainFrontendContractTests(unittest.TestCase):
    def test_train_page_controls(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        js = (ROOT / "static" / "app.js").read_text("utf-8")
        self.assertIn('id="page-train"', html)
        self.assertIn('id="trainEpochs"', html)
        self.assertNotIn('id="trainResumeEpochs"', html)
        self.assertNotIn('id="trainStudentCkpt"', html)
        self.assertNotIn('id="trainSteps"', html)
        self.assertIn("混合训练", html)
        self.assertIn("function enterTrainMixMode()", js)
        self.assertIn("/api/train/mix", js)
        self.assertNotIn("新增技能训练", html)
        self.assertNotIn("保存为任务卡片", html)
        self.assertNotIn("function openNewTrainTaskModal()", js)
        self.assertIn("已停止 ${job?.host_id||''} · ${id}（已保留 VLM cache）", js)
        self.assertNotIn("训练默认用第一个", html)
        self.assertNotIn("训练默认用第一个", js)
        self.assertIn("resolveTrainJobRefPaths", js)
        self.assertIn("pack:true", js.replace(" ", ""))
        self.assertIn("isTrainMixCompose", js)
        self.assertIn("collectMixSlotsFromUI", js)
        self.assertIn('id="trainMixSlots"', html)
        self.assertIn("mix_slots", js)
        self.assertNotIn("$('trainOutDir').value=task.train_out_dir", js)
        self.assertNotIn("body.out_dir=$('trainOutDir')", js.replace(" ", ""))
        self.assertIn("Phi_0_model_zoo", html)
        self.assertIn("Phi_0_model_zoo", js)
        self.assertIn("Phi_0_train_data", html)
        self.assertIn("Phi_0_train_data", js)
        self.assertNotIn("训完自动删包", js)
        self.assertIn("<skill>_<时间>", html)
        self.assertIn("fillTrainHostSelect", js)
        self.assertIn("host_id:hostId", js.replace(" ", ""))
        self.assertIn("/api/hosts/", js)
        self.assertIn("训练主机", html)
        self.assertIn("分技能 pack", js)

    def test_default_train_pack_base(self):
        self.assertEqual(
            train_backend.DEFAULT_TRAIN_PACK_BASE,
            "/mnt/data2/wpy/workspace/Phi_0_model_zoo/Phi_0_train_data",
        )
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "packs"
            first = train_backend.allocate_train_pack_dir(
                pack_base=str(base), skill_key="skill4", stamp="20260914_160000",
            )
            self.assertEqual(first, f"{base}/skill4_20260914_160000")
            Path(first).mkdir(parents=True)
            second = train_backend.allocate_train_pack_dir(
                pack_base=str(base), skill_key="skill4", stamp="20260914_160000",
            )
            self.assertEqual(second, f"{base}/skill4_20260914_160000_2")

    def test_allocate_train_out_dir_never_reuses_occupied(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "zoo"
            old = Path(td) / "830mix_resume16k"
            old.mkdir(parents=True)
            (old / "phi0_student_last.pt").write_bytes(b"x")
            allocated = train_backend.allocate_train_out_dir(
                out_base=str(base),
                skill_key="skill4",
                stamp="20260914_160000",
                requested=str(old),
            )
            self.assertEqual(allocated, f"{base}/skill4_20260914_160000")
            empty = Path(td) / "brand_new"
            fresh = train_backend.allocate_train_out_dir(
                out_base=str(base),
                skill_key="skill4",
                stamp="20260914_160001",
                requested=str(empty),
            )
            self.assertEqual(fresh, str(empty))
            auto = train_backend.allocate_train_out_dir(
                out_base=str(base),
                skill_key="skill4",
                stamp="20260914_160002",
                requested="",
            )
            self.assertEqual(auto, f"{base}/skill4_20260914_160002")

    def test_default_train_out_base_is_model_zoo(self):
        self.assertEqual(
            train_backend.DEFAULT_TRAIN_OUT_BASE,
            "/mnt/data2/wpy/workspace/Phi_0_model_zoo",
        )
        self.assertEqual(
            app.default_state()["infer_config"]["remote_ckpt_base"],
            train_backend.DEFAULT_TRAIN_OUT_BASE,
        )


class TrainSelectionPackTests(unittest.TestCase):
    def test_valid_only_manifest_excludes_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skill = root / "skillX"
            skill.mkdir()
            s1 = skill / "2026-09-11-19-46-45"
            s2 = skill / "2026-09-11-22-05-13"
            s1.mkdir(); s2.mkdir()
            (skill / "skillX.json").write_text(json.dumps({
                "2026-09-11-19-46-45": {"valid": [0, 2], "invalid": [1]},
                "2026-09-11-22-05-13": {"valid": [3], "invalid": [0, 1]},
            }), encoding="utf-8")
            built = train_backend.build_valid_only_manifest([str(s1), str(s2)])
            self.assertEqual(built["valid_count"], 3)
            self.assertEqual(built["invalid_count"], 3)
            self.assertEqual(built["manifest"]["2026-09-11-19-46-45"]["valid"], [0, 2])
            cls = train_backend.classify_train_selection(
                [str(s1), str(s2)], train_profile="vision_teleop"
            )
            self.assertTrue(cls["needs_pack"])
            self.assertEqual(cls["raw_root"], str(skill))
            plan = train_backend.plan_selection_pack(
                [str(s1), str(s2)], skill_id="skillX", prompt="hi", stamp="t", job_id="train_test01",
                pack_base=td,
            )
            self.assertEqual(plan["valid_count"], 3)
            self.assertFalse(plan.get("ephemeral"))
            self.assertEqual(plan["pack_root"], f"{td}/skillX_t")
            self.assertTrue(Path(plan["manifest_path"]).is_file())
            man = json.loads(Path(plan["manifest_path"]).read_text("utf-8"))
            self.assertEqual(sorted(man.keys()), ["2026-09-11-19-46-45", "2026-09-11-22-05-13"])
            # Persistent packs are kept; cleanup is a no-op.
            root = Path(plan["pack_root"])
            (root / "unified").mkdir(parents=True, exist_ok=True)
            (root / "unified" / "meta").mkdir(parents=True, exist_ok=True)
            (root / "unified" / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            cleaned = train_backend.cleanup_ephemeral_pack(plan)
            self.assertTrue(cleaned["ok"], cleaned)
            self.assertIn("not_ephemeral", cleaned.get("skipped") or [])
            self.assertTrue(root.exists())
            env = train_backend.build_train_env(
                {
                    "ref_root": plan["ref_root"],
                    "out_dir": "/tmp/out",
                    "log_file": "/tmp/out.log",
                    "stamp": "t",
                    "train_profile": "vision_teleop",
                    "pack": plan,
                    "params": {
                        "ngpu": 8, "num_envs": 32, "horizon": 32,
                        "epochs": 2, "ckpt_every": 1000, "prompt": "hi",
                    },
                },
                train_backend.load_train_catalog(),
            )
            self.assertEqual(env["MANIFEST"], plan["manifest_path"])
            self.assertTrue(env["RAW_ROOT"].endswith("/raw_stage"))
            self.assertEqual(env["REF_ROOT"], plan["ws_link"])
            self.assertEqual(env["TASK_PROMPT"], "hi")
            self.assertEqual(env["SKIP_PACK"], "0")
            self.assertEqual(cls["raw_root"], str(skill))

    def test_multi_unified_rejected(self):
        with self.assertRaises(ValueError):
            train_backend.classify_train_selection(
                [
                    "/mnt/data/a_unified",
                    "/mnt/data/b_unified",
                ],
                train_profile="vision_teleop",
            )


    def test_multi_skill_folders_staged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "skill5"
            b = root / "skill_5_throw_the_rubbish"
            a.mkdir(); b.mkdir()
            s1 = a / "2026-09-11-19-46-45"
            s2 = b / "2026-09-10-12-20-55"
            s1.mkdir(); s2.mkdir()
            (a / "skill5.json").write_text(json.dumps({
                "2026-09-11-19-46-45": {"valid": [0, 1], "invalid": [2]},
            }), encoding="utf-8")
            (b / "skill5.json").write_text(json.dumps({
                "2026-09-10-12-20-55": {"valid": [3], "invalid": []},
            }), encoding="utf-8")
            # Selecting parent folders expands to child sessions.
            cls = train_backend.classify_train_selection(
                [str(a), str(b)], train_profile="vision_teleop"
            )
            self.assertTrue(cls["needs_pack"])
            self.assertTrue(cls["multi_root"])
            self.assertEqual(len(cls["raw_sessions"]), 2)
            plan = train_backend.plan_selection_pack(
                [str(a), str(b)], skill_id="skill5", prompt="hi", stamp="t", job_id="train_multi01",
                pack_base=td,
            )
            self.assertEqual(plan["valid_count"], 3)
            self.assertTrue(plan["multi_root"])
            self.assertFalse(plan.get("ephemeral"))
            self.assertEqual(plan["pack_root"], f"{td}/skill5_t")
            stage = Path(plan["raw_root"])
            self.assertTrue((stage / "2026-09-11-19-46-45").is_symlink())
            self.assertTrue((stage / "2026-09-10-12-20-55").is_symlink())
            cleaned = train_backend.cleanup_ephemeral_pack(plan)
            self.assertTrue(cleaned["ok"], cleaned)
            self.assertIn("not_ephemeral", cleaned.get("skipped") or [])
            self.assertTrue(Path(plan["pack_root"]).exists())

    def test_plan_mix_slots_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "skillA"
            b = root / "skillB"
            a.mkdir(); b.mkdir()
            s1 = a / "2026-09-11-19-46-45"
            s2 = b / "2026-09-10-12-20-55"
            s1.mkdir(); s2.mkdir()
            (a / "skillA.json").write_text(json.dumps({
                "2026-09-11-19-46-45": {"valid": [0, 1], "invalid": [2]},
            }), encoding="utf-8")
            (b / "skillB.json").write_text(json.dumps({
                "2026-09-10-12-20-55": {"valid": [3], "invalid": []},
            }), encoding="utf-8")
            plan = train_backend.plan_mix_slots_pack(
                [
                    {"skill_id": "skillA", "prompt": "do A", "dataset_paths": [str(s1)]},
                    {"skill_id": "skillB", "prompt": "do B", "dataset_paths": [str(s2)]},
                ],
                stamp="tmix",
                job_id="train_mix01",
                pack_base=td,
            )
            self.assertTrue(plan["mix_slots_mode"])
            self.assertFalse(plan.get("ephemeral"))
            self.assertEqual(plan["valid_count"], 3)
            self.assertEqual(plan["session_count"], 2)
            self.assertEqual(len(plan["slots"]), 2)
            self.assertEqual(plan["slots"][0]["prompt"], "do A")
            self.assertEqual(plan["slots"][1]["prompt"], "do B")
            self.assertTrue(plan["pack_root"].endswith("/mix_tmix") or "/mix_tmix" in plan["pack_root"])
            self.assertIn("studio_tmp_unified_link", plan["ref_root"])
            env = train_backend.build_train_env(
                {
                    "ref_root": plan["ref_root"],
                    "out_dir": "/tmp/out_mix",
                    "log_file": "/tmp/out_mix.log",
                    "stamp": "tmix",
                    "train_profile": "mix_vision_isaac",
                    "pack": plan,
                    "params": {
                        "ngpu": 8, "num_envs": 32, "horizon": 32,
                        "epochs": 2, "ckpt_every": 1000, "prompt": "x",
                    },
                },
                train_backend.load_train_catalog(),
            )
            self.assertIn("MIX_SLOTS_JSON", env)
            self.assertEqual(env["MERGED_OUT"], plan["merged_out"])
            self.assertEqual(env["WS_LINK"], plan["ws_link"])


class TrainKillBashTests(unittest.TestCase):
    def test_force_kill_bash_includes_sigkill_and_out_dir(self):
        out = "/mnt/data3/wpy/skill1_walk_h32_b32_ddp8_e10_20260910_162439"
        bash = train_backend.build_tmux_kill_bash(
            "st_train_abcd1234", out_dir=out, job_id="train_abcd1234",
        )
        self.assertIn("kill -KILL", bash)
        self.assertIn("kill -9", bash)
        self.assertIn("_studio_kill_pat", bash)
        self.assertIn(out, bash)
        self.assertIn("train_abcd1234", bash)
        self.assertIn("tmux kill-session", bash)


if __name__ == "__main__":
    unittest.main()
