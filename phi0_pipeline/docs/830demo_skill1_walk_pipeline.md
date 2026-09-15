# 830demo skill1 walk — 训练与推理流水线

## 总览

```
原始数采 (LeRobot)          unified 512D 数据集           VLM frame cache              Student 蒸馏              MuJoCo 闭环推理
skill_1_walk_to_black_box  → 830demo_skill1_walk_unified → vlm_frame_latents_qwen3vl_dual → phi0_student_last.pt → student_sonic_dex3_cl.mp4
     + skill_1.json              (NVMe + ws symlink)         (8 GPU, CPU decode)          (8 GPU DDP, nosim)
```

**关键设计**：不是一条 `nohup` 到底，而是三步独立成功后再训；Step 1 内嵌的 VLM cache 曾因 GPU decode 太慢失败，改为 Step 2 单独用 CPU decode 重跑。

---

## Step 1 — Pack（成功）

**脚本**：`tools/data/run_830_skill2_pico_pack_and_cache.sh`  
**Python 核心**：`pack_teleop_qpos_unified_lerobot.py`

### 做什么

1. 读取 `RAW_ROOT` 下多 session 的 LeRobot 数据 + `MANIFEST`（`skill_1.json` 格式的 valid episode 列表）
2. 打包为 Phi0 **unified 512D** LeRobot 数据集（`OUT_DIR`）：
   - 手型：`dex3`
   - 动作 token：`action.motion_token` → `[396:460)`（sonic v1.1）
   - 双路视频：`ego_view` + `left_wrist`（chest_forward 别名）
   - 任务 prompt 写入 meta
3. 写 `meta/vision_episode_allowlist.json` 和 `meta/all_episode_allowlist.json`
4. （可选）lang latent cache → `meta/lang_latents_qwen3vl/`
5. **rsync** unified 到 NVMe（`NVME_DIR`），并在 workspace 建 symlink（`WS_LINK`）
6. 脚本末尾会调用 Step 2 的 VLM cache — **830 demo 中此步失败**，应用 `SKIP_PACK=1` 跳过 pack 后单独跑 Step 2

### 830 demo 实际命令

```bash
RAW_ROOT=/mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box \
MANIFEST=/mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box/skill_1.json \
OUT_DIR=/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830/830demo_skill1_walk_unified \
TASK_PROMPT='机器人朝黑箱子走过去。' \
NVME_DIR=/mnt/data3/wpy/datasets/830/830demo_skill1_walk_unified \
WS_LINK=/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified \
LOG_DIR=/mnt/data2/wpy/workspace/logs/830demo_skill1_pack_cache_20260901_093936 \
bash /mnt/data2/wpy/workspace/Phi_0_wpy/tools/data/run_830_skill2_pico_pack_and_cache.sh
```

### 输入 / 输出

| 变量 | 830 demo 值 | 说明 |
|------|-------------|------|
| `RAW_ROOT` | `.../830demo/skill_1_walk_to_black_box` | 原始数采根目录 |
| `MANIFEST` | `.../skill_1.json` | valid episode 清单（与 Studio 任务 JSON 同构） |
| `OUT_DIR` | EFS/workspace unified 输出 | pack 主输出 |
| `NVME_DIR` | `/mnt/data3/wpy/datasets/830/...` | 本地 NVMe 副本（训练读此路径） |
| `WS_LINK` | `local_nvme/datasets/830/...` | workspace 软链，方便脚本引用 |

---

## Step 2 — Dual VLM Frame Cache（成功，CPU decode）

**脚本**：`tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh`  
**Python 核心**：`cache_dual_vlm_frame_latents.py`

### 做什么

1. `--init-only` 写 cache meta
2. 启动 **8 个 GPU shard**（`CUDA_VISIBLE_DEVICES=0..7`），每个进程处理 episode 子集
3. 每帧：CPU decode 双路 mp4 → Qwen3-VL encode → 写入  
   `meta/vlm_frame_latents_qwen3vl_dual/ep/{episode:06d}/`
4. `RESUME=1` 支持断点续跑

### 830 demo 实际命令（与 Step 1 分离重跑）

```bash
DATASET_ROOT=/mnt/data3/wpy/datasets/830/830demo_skill1_walk_unified \
PROMPT='机器人朝黑箱子走过去。' \
BATCH_SIZE=32 DECODE_DEVICE=cpu DECODE_WORKERS=0 DECODE_BATCH=64 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 \
RESUME=1 \
LOG_DIR=/mnt/data2/wpy/workspace/logs/830demo_skill1_vlm_cache_resume2_20260901_093936 \
bash /mnt/data2/wpy/workspace/Phi_0_wpy/tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh
```

### 关键参数

| 参数 | 值 | 原因 |
|------|-----|------|
| `DECODE_DEVICE=cpu` | CPU 解码 mp4 | GPU NVDEC 在 EFS NFS 上太慢导致 Step 1 内嵌 cache 失败 |
| `DECODE_WORKERS=0` | 自动 | RefVideoFrameSource 在 CPU decode 时用 16 workers |
| `BATCH_SIZE=32` | VLM encode batch | 8 GPU 并行 |
| `RESUME=1` | 断点续跑 | 长跑任务必备 |

### 训练前置检查文件

```
${REF_ROOT}/meta/stats.json
${REF_ROOT}/meta/vision_episode_allowlist.json
${REF_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/meta.json
```

---

## Step 3 — 默认 nosim 训练（成功，e10 / 15290 step）

**脚本**：`tools/train/run_830_walk_blackbox_vlm_cache_distill.sh`  
**实际入口**：`run_online_vlm_mix_distill.sh`（通用 online VLM 蒸馏 launcher）

### 做什么

1. 设置 **纯 vision DataLoader** 模式（不启动 Isaac Sim）：
   - `PHI0_VISION_DL_ONLY=1` — 跳过 8× Isaac 启动
   - `PHI0_USE_VLM_FRAME_LATENT_CACHE=1` — 读 Step 2 的 frame cache，不再在线 decode 视频
   - `PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1`
   - `VISION_ONLY=1` — 只用 vision_episode_allowlist
2. Student：ChunkStudent BC + DAgger（β=0.8）
3. 部署域：`sonic_v1_1` + `dex3` 手
4. 默认：`NGPU=8`, `NUM_ENVS=32`, `HORIZON=32`, `EPOCHS=10`

### 830 demo 实际命令

```bash
REF_ROOT=/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified \
PHI0_DISTILL_OUT=/mnt/data3/wpy/830demo_skill1_walk_vlm_cache_h32_b32_ddp8_e10_20260901_093936 \
LOG_FILE=/mnt/data2/wpy/workspace/Phi_0_wpy/logs/830demo_skill1_walk_vlm_cache_h32_b32_ddp8_20260901_093936.log \
STAMP=20260901_093936 \
bash /mnt/data2/wpy/workspace/Phi_0_wpy/tools/train/run_830_walk_blackbox_vlm_cache_distill.sh
```

### 可选：继续训 50k step（后续加训，非最初 e10）

```bash
STUDENT_CKPT=.../e10_.../phi0_student_last.pt EXTRA_STEPS=50000 \
REF_ROOT=.../830demo_skill1_walk_unified \
PHI0_DISTILL_OUT=.../resume50k_... CKPT_EVERY=2000 PHI0_CKPT_STEP_KEEP=0 \
bash .../run_830_walk_blackbox_vlm_cache_distill.sh
```

---

## Step 4 — 推理（MuJoCo 闭环）

**脚本**：`tools/eval/run_830_walk_student_cl_mujoco_viz.sh`  
**底层**：`run_sonic_latent_sim_eval.sh`（ZMQ v4 → deploy → MuJoCo 录 mp4）

### 做什么

1. 从 unified parquet 读 episode 帧数、计算 `REF_START` 偏移
2. 加载 `STUDENT_CKPT`，闭环控制 MuJoCo 中的 G1
3. VLM：从 **frame cache** 读 latent（`PHI0_CL_VLM_SOURCE=frame_cache`），与训练对齐
4. 手观测：`PHI0_CL_HAND_OBS=commanded`（与训练 hand_ref 语义一致）
5. 输出：`${OUT}/student_sonic_dex3_cl.mp4`

### 前置：valid-hand 软链

推理 episode 0 需要 valid hand parquet（out ep0 → raw source ep1）：

```bash
VH=/tmp/830demo_skill1_valid_hand_ep0
mkdir -p "$VH/data/chunk-000"
ln -sfn /mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box/2026-09-01-17-00-42/data/chunk-000/episode_000001.parquet \
  "$VH/data/chunk-000/episode_000000.parquet"
```

### 830 demo 实际命令

```bash
export STUDENT_CKPT=/mnt/data3/wpy/830demo_skill1_walk_vlm_cache_h32_b32_ddp8_resume50k_20260901_105405/phi0_student_last.pt
export REF_ROOT=/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified
export EP=0
export HORIZON=32
export TAG=830demo_skill1_cl_ep0_s65290_20260901_223208
export OUT=/mnt/data2/wpy/workspace/Phi_0_wpy/experiments/830demo_skill1_cl_ep0_s65290_20260901_223208
export CUDA_VISIBLE_DEVICES=4
export PHI0_PY=/mnt/data3/wpy/conda-envs/Phi-0-wbc-wpy/bin/python
export VALID_HAND_ROOT=/tmp/830demo_skill1_valid_hand_ep0
export DEPLOY_POLICY_DIR=sonic_v1_1
export PHI0_HAND_MODE=dex3
export PHI0_NEWTON_REVO2=0
export PHI0_USE_VLM_FRAME_LATENT_CACHE=1
export PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1
export PHI0_CL_VLM_SOURCE=frame_cache
export PHI0_CL_HAND_OBS=commanded
export PHI0_ADALN_ZERO_VISION=1
export PHI0_VLM_ATTN=flash_attention_2
export PHI0_ALLOW_SDPA_FALLBACK=0
bash /mnt/data2/wpy/workspace/Phi_0_wpy/tools/eval/run_830_walk_student_cl_mujoco_viz.sh
```

---

## 数据流与存储位置

```
/mnt/data2/wpy/workspace/830demo/skill_1_walk_to_black_box/   ← 原始数采（Studio 可管理）
/mnt/data2/wpy/workspace/Phi0_Dataset/.../830demo_skill1_walk_unified/  ← Pack EFS 输出
/mnt/data3/wpy/datasets/830/830demo_skill1_walk_unified/      ← NVMe 训练副本
/mnt/data2/wpy/workspace/local_nvme/datasets/830/...          ← symlink → NVMe
/mnt/data3/wpy/830demo_skill1_walk_vlm_cache_h32_b32_ddp8_e10_*/  ← checkpoint
/mnt/data2/wpy/workspace/Phi_0_wpy/experiments/830demo_skill1_cl_*/  ← 推理 mp4
```

---

## Web UI 集成建议（对照 template.html）

建议新增侧边栏 **04 训练 · Train**、**05 推理 · Eval**，每步对应 manifest 中的一个 `stage`：

1. **表单**：预填 env 变量（任务 prompt、路径、GPU 数等），支持从已注册 Studio 任务自动填充 `RAW_ROOT` / `MANIFEST`
2. **依赖检查**：Step 3 前 SSH 检查 `meta.json` 等文件是否存在
3. **执行**：复用现有 `conversion_worker` 模式 — SSH + `env ... bash script` + 日志流
4. **状态机**：`pending → running → completed / error`，支持 `RESUME` 步骤重跑
5. **向导 UI**：参考 `template.html` 的 Step N + 进度点 + 摘要页

后端可新增 `/api/pipelines` 读取 `pipeline_manifest.json`，`/api/pipelines/<id>/run` 启动远端任务。
