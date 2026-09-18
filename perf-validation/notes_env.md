# PR #1 性能验证 — 中间记录（截至 SIGKILL blocker）

## 1. 仓库 / PR 状态（已验证）

- origin/main = `225d74e`（本地 main 同步）
- perf/macos-m4-16gb = `e4cc70b`（PR #1 head，4 commits：f8277d5, 9c1d19d, 45ab301, e4cc70b）
- PR #1：OPEN / MERGEABLE / +407 -19 / 4 files（.env.macos.example, docs/zh/MACOS_M4_PERFORMANCE.md, run_api_macos.sh, scripts/macos_perf_check.py）
- 本地 perf 分支已建（e4cc70b），工作区 clean
- 网络注：git https 直连 github.com 不通（curl 443 超时），gh API 通道可用（gh pr checkout 成功取到分支）

## 2. 机器信息

- 芯片：Apple M4（Mac16,10 = Mac mini M4）
- 内存：16 GiB 统一内存
- macOS：26.4.1 (25E253)
- venv Python：3.11.14 ｜ uv：0.9.27（/opt/homebrew）
- torch：2.10.0（MPS available/built）

## 3. 预检脚本（已验证）

- `cp .env.macos.example .env` 后 `.venv/bin/python scripts/macos_perf_check.py` 输出正常：
  - 16.0 GiB / arm64 / swap total 2048M used 624M / memory free 74% / API not running
  - profile 各键显示正确，"Profile looks consistent"，exit 0

## 4. 启动器 / 配置生效性（源码+日志双重确认）

- `.env` 被 run_api_macos.sh source（日志 profile 正确）：LM=mlx/0.6B、MLX DiT=false、MLX VAE=0 (chunk=192)、CPU offload=false、worker=1、cache root=`~/Library/Caches/ace-step`
- **无 /Users/anshi 硬编码**（diff 确认旧硬编码全部移除，改为 `$HOME`/`$SCRIPT_DIR`/`ACESTEP_CACHE_ROOT`）
- 疑点1：日志 "Auto-enabling CPU offload (GPU < 16GB)" 为无条件打印（startup_model_init.py:42 区域），实际 `ACESTEP_OFFLOAD_TO_CPU=false` 被代码尊重（env 显式设置时优先）→ 行为正确，日志误导
- 疑点2：`ACESTEP_MLX_VAE=0` 时 decode 走 MPS tiled（vae_decode.py 检查 env）✓，但 `_init_mlx_vae()` 在 mps 设备无条件执行（init_service_setup.py:140）→ 启动时仍预编译 MLX VAE（白占内存，与文档"PyTorch/MPS tiled"表述部分冲突）
- flash attention → SDPA 降级（MPS 已知限制，WARNING）
- DiT 从项目 checkpoints 加载一次（无重复加载日志）

## 5. BLOCKER：启动 SIGKILL（exit 137）— 根因已定位

- 两次启动均死在 **MLX LM 权重加载阶段**（"Loading MLX model from ..." 之后无输出，进程 exit 137=SIGKILL）
- **根因（已定位）：系统盘 `/System/Volumes/Data` 仅剩 3.2GB 可用（228Gi 用 196Gi，99% 满）→ swap 文件无法扩展 → 模型加载峰值内存不足时 swap 耗尽 → jetsam 杀进程**
- 佐证：swap total 从 2048M 动态扩到 5120-6144M 后无法再扩；两次死亡时 swap used 均 ~4.3-4.7G
- 辅助因素：系统总 RSS 8.8GB（Doubao 1.4G / ChatGPT+Codex 1.05G / Xiaomi 0.23G / 其余分散）；内存竞争加剧
- 判定：环境（磁盘满 + 内存竞争）导致，非 PR 代码逻辑错误（先前 v1-v3 内存更空闲时可启动）
- 空间调查：~/Library/Caches 8.0G、~/Downloads 1.2G、~/.cache 2.2G、项目目录 5.9G；SSD 716G 空闲（模型在 SSD，无问题）
- 处置：请求用户释放系统盘 ≥8GB（如清理缓存/废纸篓）+ 关闭 ChatGPT/MiMo/Chrome，再重试

## 6. 待办（内存释放后）

- [ ] 启动成功：记录冷启动/模型加载时间、启动后 RSS、memory pressure、swap、MPS fallback、是否重复加载
- [ ] benchmark 20/30/60/90s × 2（bench_client.py 已就绪，端点 /release_task + /query_result，响应含 time_costs）
- [ ] 连续多 seed：30s×4、60s×3
- [ ] 对照 B（ACESTEP_USE_MLX_DIT=true）：20/30s
- [ ] 可选 C（offload=true）：30s
- [ ] SSD cache（ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step）：首次加载/启动对比
- [ ] 汇总 A-J 报告
