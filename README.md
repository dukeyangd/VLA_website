# WB-VLA Data-station（cluster_0）

本机路径：`/mnt/data2/gzy/workspace/wb-vla-data-station-src`  
面向 **02 筛选 → 03 训练 → 04 推理**（01 采集在其它机器，本机不配）。

## 关键路径

| 项 | 路径 |
|----|------|
| Studio | `/mnt/data2/gzy/workspace/wb-vla-data-station-src` |
| Phi0 权威树 | `/mnt/data2/wpy/workspace/Phi_0_wpy` |
| Phi0 Python | `/home/neotix/miniforge3/envs/Phi-0-wpy/bin/python` |
| 830 unified | `/mnt/data2/wpy/workspace/local_nvme/datasets/830/` |
| raw / 技能数据 | `/mnt/data2/wpy/workspace/830demo/` |
| ckpt | `/mnt/data3/wpy/` |
| 训练入口 | `Phi_0_wpy/tools/train/run_online_vlm_mix_distill.sh` |
| 训练 wrapper | `Phi_0_wpy/tools/train/run_830_*_vlm_cache_distill.sh` |
| 推理编排 | `Phi_0_wpy/tools/eval/studio_cl_orchestrator.sh` |
| 推理引擎 | `Phi_0_wpy/tools/eval/run_sonic_latent_sim_eval.sh` |
| 最新学生权重示例 | `/mnt/data3/wpy/830mix_skill1234_demo5_handcmd_lag1_vlm_cache_h32_b32_ddp8_e2_pv0.95_resume16k_20260909_011924/phi0_student_last.pt` |
| UI catalog | `phi0_pipeline/train_catalog.json` · `infer_catalog.json` |
| 技能卡示例 | `data/skills/` |

## 设计（简）

Flask 只做编排与状态；GPU 训练/推理在 `Phi_0_wpy` 执行。  
本机 `Host cluster_0` → `127.0.0.1`（回环 SSH），网页里仍选 `cluster_0`。  
**03 训练** 在远端/本机以 **tmux 会话** 托管：Studio / 网页崩溃不会带走训练；重启网页后仍可读日志与曲线。

## 技能卡数据链路

```text
技能卡(skill) ──绑定──► prompt / host / remote_dir / REF / ckpt / hand_obs
        │
        ├─ 02 筛选：session 或 unified 上打 valid/invalid → skill_N.json / labels
        ├─ 03 训练：卡片任务 + *_unified(REF+VLM cache) → distill → /mnt/data3/.../phi0_student_last.pt
        └─ 04 推理：卡片 ckpt + ref_root → orchestrator（y → ] → Enter）
```

全链路共用当前选中技能卡；换卡即换数据与权重配置。

## 操作

```bash
cd /mnt/data2/gzy/workspace/wb-vla-data-station-src
bash start.sh      # http://127.0.0.1:7890
# 停：fuser -k 7890/tcp
# 端口占用可重跑上面命令，或 PORT=7891 bash scripts/start_on_cluster.sh
```

网页：选技能卡 → **02** 打标/导出 → **03** Host=`cluster_0`、`phi0_root`=`/mnt/data2/wpy/workspace/Phi_0_wpy`、选 unified REF → **04** 填 ckpt/REF、Deploy Terminal 闸门。

---

## 03 训练

### 网页流程

1. 选技能卡（skill1–6；**skill2 = `skill_2_pico_pick_and_place_pure2`**，忽略 `skill_2_pico_pick_the_toy`）。
2. 数据集：
   - **多选 raw session / 技能文件夹**（推荐）：可跨多个 830demo 目录（如 `skill5` + `skill_5_throw_the_rubbish`）。合并全部勾选 session 的 **valid** → 软链到临时 RAW_ROOT → pack 到 `/mnt/data3/wpy/tmp/studio_train_packs/<job>/` → distill；**训完/取消自动删包**。原始数据只读不动。
   - 或选一个已有 `*_unified` REF（须已经过 02 筛选；训练只读 valid allowlist）。
   - 单技能**不能**直接用 `830mix_*` 包。
3. Host=`cluster_0`，启动后看「曲线 / 终端 / ckpt」。
4. 产物目录默认 `/mnt/data3/wpy/<tag>_<stamp>/`，主权重 `phi0_student_last.pt`。

网页端蒸馏入口始终是：

`Phi_0_wpy/tools/train/run_online_vlm_mix_distill.sh`

多选 raw session 时先走：

`tools/train/run_studio_selection_pack_and_distill.sh`
（内部复用官方 `run_830_skill2_pico_pack_and_cache.sh`，再调用上面的 distill 入口）

技能差异靠 `train_profile` 自适应（不再静默回退到 skill1 walk 脚本）：

| profile | 用途 |
|---------|------|
| `vision_teleop` | skill2/3/4/6 等纯视觉 teleop |
| `vision_teleop_handcmd` | pick_toy 等 handcmd lag1 |
| `vision_teleop_handcmd_rtc` | skill1 walk |
| `mix_vision_isaac` | 已 pack 的 830mix_*_unified（含 demo5） |

**混合训练**：UI 多选技能只用于选定 mix 配方（s123 或 s1234）；REF 必须是已跑 pipeline 产出的 `830mix_*_unified`，不会现场拼包。

### tmux（网页崩溃训练不停）

- **新启动的 03 训练**会在执行机创建 detached tmux：`st_train_<jobid>`。
- 训练进程挂在 tmux 下，**与 Flask/浏览器生命周期解耦**；网页崩了、Studio 重启，训练继续跑。
- Studio 重启后会自动 `resume` 监视仍在跑的 job，网页可继续看日志 / 曲线（`distill_metrics.jsonl` 或日志进度条回退）/ ckpt。
- 手动查看 / 附着：
  ```bash
  tmux ls
  tmux attach -t st_train_<jobid>
  # 日志
  tail -f /mnt/data2/wpy/workspace/Phi_0_wpy/logs/<tag>_<stamp>.log
  ```
- 网页点「停止」会 `tmux kill-session`（结束该次训练）。
- **注意**：改代码前用旧方式（SSH 前台挂着）起的任务没有 tmux；那种模式网页/SSH 出问题时训练可能一起挂。请用网页重新点一次「开始训练」以走 tmux。

### VLM latent cache

830 系列 wrapper（`run_830_*_vlm_cache_distill.sh`）默认：

- `PHI0_USE_VLM_FRAME_LATENT_CACHE=1`
- `PHI0_VLM_FRAME_CACHE_SKIP_VIDEO=1`
- `USE_VLM=0`（训练时不跑在线 Qwen，只读预编码 cache）

**缓存路径**（相对 REF 数据集根）：

```text
{REF_ROOT}/meta/vlm_frame_latents_qwen3vl_dual/
  meta.json
  ep/{episode:06d}/
    latents.fp16.dat
    masks.uint8.dat
    lengths.i32.dat
  logs/                    # encode 时的 shard 日志
```

目录名可用环境变量 `PHI0_VLM_FRAME_LATENTS_DIRNAME` 覆盖，默认 `vlm_frame_latents_qwen3vl_dual`。

示例（本机 skill1）：

- REF 软链：`/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified`
- 实际：`/mnt/data3/wpy/datasets/830/830demo_skill1_walk_unified`
- cache：`…/meta/vlm_frame_latents_qwen3vl_dual/`（双视角 ego + left_wrist）

**没有 cache 时：训练不会自动算一轮。**  
wrapper 启动前检查 `meta.json`，缺失则 `exit 1`；训练代码在 cache 缺失时也会 `FileNotFoundError`。

需先手动 encode（写入上述 `meta/vlm_frame_latents_qwen3vl_dual/`）：

```bash
cd /mnt/data2/wpy/workspace/Phi_0_wpy
DATASET_ROOT=/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified \
PROMPT='机器人朝黑箱子走过去。' \
bash tools/data/run_cache_dual_vlm_frame_latents_8gpu.sh
```

混训 pipeline（如 `run_830mix_skill1234_demo5_pipeline.sh`）会在 pack 阶段主动调用该 encode；**纯 distill 训练脚本不会。**

### 命令行训练示例

```bash
cd /mnt/data2/wpy/workspace/Phi_0_wpy
REF_ROOT=/mnt/data2/wpy/workspace/local_nvme/datasets/830/830demo_skill1_walk_unified \
PHI0_DISTILL_OUT=/mnt/data3/wpy/my_run_$(date +%Y%m%d_%H%M%S) \
bash tools/train/run_830_skill1_walk_vlm_cache_distill.sh
```


## ziyi密（不要改动，原样保留）
fuser -k 7890/tcp
