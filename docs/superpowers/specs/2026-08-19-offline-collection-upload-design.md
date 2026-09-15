# 离线数采、实时打标与联网补传设计

## 目标

将数据采集与远端任务完全解耦。无网数采阶段只需要一个本地数据目录，Studio 持续扫描并实时展示 Episode，用户可以标记 `valid` / `invalid`。结束数采时在该目录根部生成 `labels.json`。恢复网络后，用户注册远端任务，再将这次会话的原始数据与 `labels.json` 上传到指定 session 目录，并把标注原子合并到远端任务级 JSON。

## 范围

本次实现包括：

- 无任务、无 Host 依赖的离线数采会话。
- 采集中持续扫描本地目录并保留已有标注。
- 每次标注及结束数采时写入 `<local_dir>/labels.json`。
- 联网后注册任务、选择待上传会话并执行一次补传。
- 将 session 标注合并到远端任务根目录的任务级 JSON。
- 明确的离线、待上传、上传中、完成和失败状态，以及失败重试。

本次不改变远端 512D 转换流程，不自动检测网络切换，不删除本地原始数据，也不提供断点队列调度器以外的自动上传策略。

## 用户流程

### 1. 离线数采

用户只输入本地数据目录并点击“开始离线数采”。后端创建目录（若不存在），建立离线 collection，并启动只读扫描线程。该过程不得调用 SSH、Host 测试、远端建目录或 rsync。

扫描线程约每 3 秒调用一次本地扫描。页面现有的 5 秒状态刷新会展示新增文件、数据大小和 Episode。用户可在数据打标页实时选择新 Episode 并标记。

### 2. 本地实时标注

每次标记 `valid`、`invalid` 或恢复为 `unreviewed` 时，后端先更新 collection 状态，再原子写入 `<local_dir>/labels.json`。写入失败时接口返回错误，且不得声称标注已经持久化。

重复扫描必须通过稳定身份（优先使用相对路径）把旧标注迁移到新的 Episode 列表，避免新增文件导致排序变化后标注错位或丢失。`labels.json` 本身不作为可标注 Episode，也不计入原始数据文件统计。

### 3. 结束数采

点击“结束数采”后，后端通知扫描线程停止。线程执行最后一次本地扫描和 `labels.json` 原子写入，然后将状态设置为 `pending_upload`。结束过程不访问网络。

若服务在采集中重启，原 `collecting_offline` / `stopping_offline` 会话标记为 `interrupted`；用户可以重新扫描、继续标注，并通过结束操作生成最终文件。已存在的 `labels.json` 在重新载入/扫描时用于恢复标注。

### 4. 联网注册任务

恢复网络后，用户沿用远端任务注册表单。此时才测试实际远端操作：创建 `<remote_base>/<task_name>`，失败则不注册任务。

离线 collection 不需要也不允许在创建时绑定 `task_id`、`host_id` 或 `remote_dir`。这些字段在开始补传时写入。

### 5. 补传

对 `pending_upload`、`upload_error` 或可恢复的 `interrupted` 会话点击“上传”，在弹窗中选择：

- 已注册任务；
- 远端 Host（默认任务 Host）；
- 远端 session 目录（默认 `<task.remote_dir>/<local_dir basename>`）。

后端验证目标是任务根目录的直接或更深子目录，创建远端 session 目录，并执行一次 rsync：

```text
<local_dir>/  ->  <host>:<remote_session_dir>/
```

上传包含原始数据和 `labels.json`，仅排除 Studio 的内部目录（如仍存在 `.humanoid_data/`）。rsync 成功后，后端从本地 `labels.json` 构造任务级 entry，并通过现有带文件锁和原子替换的远端合并脚本写入：

```text
<task.remote_dir>/<manifest_basename(task.name)>
```

键名使用远端 session 目录 basename。只有“rsync 成功 + 远端任务 JSON 合并成功”才将状态设为 `uploaded`。任一步失败都设为 `upload_error`，保留本地文件、选择的远端目标和日志；再次点击上传可增量重试。

## 数据模型

### Collection

离线会话创建时包含：

```json
{
  "id": "collect_ab12cd34",
  "source_type": "offline",
  "local_dir": "/local/session",
  "status": "collecting_offline",
  "message": "正在扫描本地数采数据",
  "created_at": "...",
  "file_count": 0,
  "bytes_total": 0,
  "episodes": [],
  "labels_file": "/local/session/labels.json",
  "logs": []
}
```

开始补传时追加 `task_id`、`task_name`、`host_id` 和 `remote_dir`。成功后追加 `uploaded_at`、`last_sync_at` 和 `remote_manifest_file`。

状态集合：

- `collecting_offline`：后台持续扫描。
- `stopping_offline`：等待最后扫描与落盘。
- `pending_upload`：离线阶段完成，可联网补传。
- `uploading`：正在创建目录、rsync 或合并 JSON。
- `uploaded`：原数据和任务 JSON 均已成功写入远端。
- `upload_error`：补传失败，可重试。
- `interrupted`：服务重启中断了活动线程，可本地恢复。

### labels.json

本地文件采用自描述、可恢复的版本化格式：

```json
{
  "version": 1,
  "session": "session",
  "created_at": "2026-08-19T14:00:00+0800",
  "updated_at": "2026-08-19T14:30:00+0800",
  "valid": [
    {"name": "episode_000001", "relative_path": "videos/.../episode_000001.mp4", "episode_number": 1}
  ],
  "invalid": [],
  "unreviewed": [],
  "counts": {"valid": 1, "invalid": 0, "unreviewed": 0}
}
```

数组保存对象而不只保存编号，以便页面从文件恢复标注并处理没有数字后缀的 Episode。上传到当前远端任务 JSON 时，维持现有转换链路格式：

```json
{
  "session": {
    "valid_count": 1,
    "valid": [1],
    "invalid": []
  }
}
```

没有可解析数字后缀的已标注 Episode 会保留在本地 `labels.json`，但远端旧格式无法表达；上传前应阻止操作并给出明确错误，而不是静默丢弃。

## 后端接口

- `POST /api/collections/start`：请求仅需 `local_dir`；创建离线会话，不启动 rsync。
- `POST /api/collections/<id>/stop`：停止本地扫描，最终写入 `labels.json`。
- `POST /api/collections/<id>/mark`：只更新本地 collection 和 `labels.json`，绝不访问远端。
- `POST /api/collections/<id>/refresh`：扫描并保留/恢复标注，再写本地 JSON。
- `POST /api/collections/<id>/upload`：接收 `task_id`、`host_id`、`remote_dir`，启动一次性补传 worker。
- `GET /api/collections/<id>/manifest`：对离线会话返回 `labels.json` 的路径和内容。

现有 `/api/hosts/test` 保留为用户主动操作；任何离线 collection 接口都不会隐式调用它。

## 前端

数据采集页调整为：

1. “开始离线数采”：仅显示本地数据目录和开始按钮，文案明确“无网络、无上传”。
2. “联网后注册远端任务”：保留现有任务表单。
3. 会话列表：采集中显示“结束数采”和“去标注”；待上传显示“去标注”和“上传”；上传失败显示错误与“重试上传”；已上传显示远端目标。

上传按钮打开新弹窗，选择任务、Host、远端 session 路径。数据打标页在采集中保持可用，页面定时刷新时不强制切走当前 Episode；新增 Episode 出现在列表中。

所有旧的“实时同步”“开始数采与实时上传”“JSON 已同步”文案改成离线语义。统计项“已同步数据”改成“本地已扫描”。

## 原子性、错误处理与安全

- `labels.json` 使用同目录临时文件、flush/fsync 和 `os.replace` 原子写入。
- 本地目录必须是绝对路径；远端 session 必须在所选任务根目录之下。
- rsync 使用 `--partial` 支持失败后增量重试，并继续使用禁用 SSH forward 的连接参数。
- 上传 worker 不修改或删除本地原始数据和 `labels.json`。
- 上传请求若已有 worker 运行则拒绝重复启动。
- 远端 JSON 继续使用文件锁，支持多个 Studio 会话并发合并。
- 任务 JSON 合并失败时不回滚已上传原数据；重试只需再次增量 rsync 并重新合并。

## 兼容性与迁移

现有 collection 记录继续显示和标注。新的离线流程只为 `source_type: "offline"` 使用新状态。历史 `collected` / `imported` 项不自动改写，也不删除现有 `.humanoid_data/manifests` 文件。

旧实时同步入口从主页面移除，但后端历史数据读取保持兼容。新代码不再为新会话启动旧 `sync_worker`。

## 测试与验收

自动化测试至少覆盖：

- 开始离线会话不会调用 `ssh`、`rsync` 或 `remote_mkdir`。
- 扫描新增 Episode 时保留已标注 Episode 的状态。
- 标注后立即在数据根目录原子生成正确的 `labels.json`。
- `labels.json` 可在服务状态丢失后的扫描中恢复标注。
- 结束数采执行最终扫描并进入 `pending_upload`。
- 上传包含 `labels.json`，目标路径正确，并在成功合并后进入 `uploaded`。
- rsync 或远端合并失败进入 `upload_error`，再次请求可以重试。
- 没有数字后缀的已标注 Episode 会阻止旧格式远端合并并显示原因。
- 现有 SSH 命令继续禁用配置中的 forward。

手工验收流程：连接 `XSENS_5g`，启动会话，持续写入测试 Episode，确认网页实时出现并可标注；结束后断言根目录存在 `labels.json`；切回有网环境，注册任务并补传；最后在 cluster_0 检查 session 原数据、session 内的 `labels.json` 和任务根目录聚合 JSON。
