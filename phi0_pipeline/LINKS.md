# 本机可选软链（勿打进分享包）

仅在**本机**需要直接引用 Phi0 / GR00T 源码或仿真 venv 时再创建。路径按新机器实际位置填写。

```bash
cd phi0_pipeline
ln -s /path/to/Phi_0_wpy/src src
ln -s /path/to/Phi_0_wpy/subpackages subpackages
ln -s /path/to/GR00T-WholeBodyControl/.venv_sim .venv_sim
```

训练/推理默认走 SSH 到 cluster 上的 `Phi_0_wpy` 时，可不建这些软链。
