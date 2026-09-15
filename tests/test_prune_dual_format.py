"""Prune script dual-format detection + dry-run."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sonic_qa.prune_lerobot_v3_episodes import detect_format, prune_dataset


class PruneDualFormatTest(unittest.TestCase):
    def test_detect_v21(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta = root / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(
                json.dumps({"codebase_version": "v2.1", "total_episodes": 2}), "utf-8"
            )
            (meta / "episodes.jsonl").write_text(
                '{"episode_index":0,"length":3,"tasks":["a"]}\n'
                '{"episode_index":1,"length":4,"tasks":["a"]}\n',
                "utf-8",
            )
            self.assertEqual(detect_format(root), "v2.1")

    def test_detect_v3(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta = root / "meta" / "episodes" / "chunk-000"
            meta.mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0", "total_episodes": 1}), "utf-8"
            )
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0],
                        "length": [2],
                        "tasks": [["t"]],
                    }
                ),
                meta / "file-000.parquet",
            )
            self.assertEqual(detect_format(root), "v3")

    def test_dry_run_v21(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta = root / "meta"
            data = root / "data" / "chunk-000"
            meta.mkdir()
            data.mkdir(parents=True)
            (meta / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v2.1",
                        "total_episodes": 2,
                        "total_frames": 7,
                        "fps": 30,
                    }
                ),
                "utf-8",
            )
            (meta / "episodes.jsonl").write_text(
                '{"episode_index":0,"length":3,"tasks":["pick"]}\n'
                '{"episode_index":1,"length":4,"tasks":["pick"]}\n',
                "utf-8",
            )
            for i, n in enumerate([3, 4]):
                pq.write_table(
                    pa.table(
                        {
                            "index": list(range(n)),
                            "episode_index": [i] * n,
                            "frame_index": list(range(n)),
                        }
                    ),
                    data / f"episode_{i:06d}.parquet",
                )
            report = prune_dataset(root, {1}, dry_run=True)
            self.assertEqual(report["format"], "v2.1")
            self.assertEqual(report["kept_after"], 1)
            self.assertEqual(report["deleted"], [1])


if __name__ == "__main__":
    unittest.main()
