# Phi0 训练 / 推理流水线（cluster_0 镜像）

本目录从 `cluster_0:/mnt/data2/wpy/workspace/Phi_0_wpy/tools/` 拉取的 **830 demo skill1 walk** 相关脚本与说明，供 Humanoid Data Studio 后续网页端替代终端命令行操作。

> **执行环境**：脚本在 cluster_0 上运行；本目录为本地参考副本。实际跑任务时 SSH 到 `cluster_0`，使用远端 `Phi_0_wpy` 根目录下的同名脚本（或 `PHI0_ROOT` 指向该路径）。

## 目录结构

```
phi0_pipeline/
├── README.md                          # 本文件
├── pipeline_manifest.json             # 结构化步骤定义（供 Web UI 读取）
├── docs/
│   └── 830demo_skill1_walk_pipeline.md  # 三步训练 + 推理详解
├── examples/830demo_skill1/
│   ├── step1_pack.env                 # Pack 环境变量
│   ├── step2_vlm_cache.env            # VLM frame cache（CPU decode）
│   ├── step3_train_e10.env            # 默认 nosim 训练 e10
│   ├── step3_train_resume50k.env      # 可选：继续训 50k step
│   ├── step4_infer.env                # MuJoCo 闭环推理
│   └── valid_hand_ep0_setup.sh        # valid-hand 软链准备
└── tools/                             # 从 cluster_0 rsync 的脚本
    ├── data/
    ├── train/
    ├── eval/
    └── env/
```

## 830 demo 三步训练（已跑通）

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 1 Pack | `run_830_skill2_pico_pack_and_cache.sh` | 原始 LeRobot → unified 512D；写 allowlist；rsync 到 NVMe；**可跳过内嵌 VLM cache** |
| 2 VLM Cache | `run_cache_dual_vlm_frame_latents_8gpu.sh` | 8 GPU 分片，**CPU decode** 双视角 + prompt → `meta/vlm_frame_latents_qwen3vl_dual/` |
| 3 Train | `run_830_walk_blackbox_vlm_cache_distill.sh` | nosim vision_dl + frame cache 蒸馏，默认 8 GPU / H=32 / e10 |

推理：`run_830_walk_student_cl_mujoco_viz.sh` → `run_sonic_latent_sim_eval.sh`（MuJoCo 闭环 + mp4）。

详细逻辑见 [docs/830demo_skill1_walk_pipeline.md](docs/830demo_skill1_walk_pipeline.md)。

## 与 Humanoid Data Studio 的关系

| Studio 现有模块 | 本流水线 |
|-----------------|----------|
| 01 数据采集 / 打标 | 提供 `RAW_ROOT` + `MANIFEST`（skill_N.json） |
| 03 512D 转换 | 类似 Step 1 Pack，但 Phi0 unified 格式更完整 |
| **待建：04 训练 / 05 推理** | 读取 `pipeline_manifest.json`，分步表单 + 远端 SSH 执行 + 日志流 |

UI 参考：`template.html` 的多步向导（Step N of M）+ 现有 `static/index.html` 侧边栏布局。

## 更新脚本

```bash
SSH="ssh -F /home/neotix/.ssh/config -o BatchMode=yes -o ClearAllForwardings=yes"
REMOTE=/mnt/data2/wpy/workspace/Phi_0_wpy
BASE=/home/neotix/noetix/humanoid_data_studio/phi0_pipeline

rsync -az -e "$SSH" "cluster_0:${REMOTE}/tools/data/run_830_skill2_pico_pack_and_cache.sh" "$BASE/tools/data/"
# … 其他文件同理
```
