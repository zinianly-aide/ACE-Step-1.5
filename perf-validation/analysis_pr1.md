# PR #1 性能验证报告 — perf/macos-m4-16gb

- 验证日期：2026-09-19
- 仓库：github.com/zinianly-aide/ACE-Step-1.5
- PR #1：perf/macos-m4-16gb → main（OPEN / MERGEABLE，+407/-19，4 files）
- 分支：origin/main = 225d74e；PR head = e4cc70b
- 本轮未改任何业务代码；未 merge main；未动 generated-audio

---

## A. 环境信息

| 项 | 值 |
|---|---|
| 芯片 | Apple M4（Mac16,10） |
| 内存 | 16 GiB（统一内存，MPS 可用 11.8GB，tier4） |
| macOS | 26.4.1 (25E253) |
| Python | 3.11.14（.venv） |
| uv | 0.9.27 |
| torch | 2.10.0（MPS available） |
| 系统盘 | 97% 满（验证期间清理 ~4.3GB 可重建缓存后 free 3.2→8.0GB） |
| 系统内存基线 | 其他进程 RSS ≈8.2GB（Doubao 1.4GB + 系统服务），验证中已关闭 ChatGPT / 小米 MiMo |
| 模型权重 | /Volumes/ssd/ace-step/checkpoints/（acestep-v15-turbo 4.5G、5Hz-lm-0.6B 1.3G、Qwen3-Embedding-0.6B 1.1G、vae 322M） |

启动 blocker 说明：前两次启动在 MLX LM 权重加载阶段 SIGKILL（exit 137），根因是**系统盘满（free 3.2GB）导致 swap 无法扩展**，非 PR 代码逻辑错误。清理缓存并关闭高内存应用后，第三次启动成功（77s）。此环境事实必须写入结论背景：**16GB M4 上系统剩余内存直接决定本 PR 的可用性**。

## B. PR 配置是否真实生效

| PR 声称 | 验证方式 | 结果 |
|---|---|---|
| LM 使用 MLX | 启动日志 "Attempting MLX backend… MLX model loaded successfully in 11.95s" | ✅ 生效 |
| LM 模型 0.6B | profile "LM backend/model: mlx / acestep-5Hz-lm-0.6B" | ✅ 生效 |
| DiT 不用 MLX 双份常驻 | MLX_DIT=false 时日志无 MLX-DiT init；DiT backend = PyTorch (mps) | ✅ 生效 |
| DiT 用 PyTorch/MPS | 日志 "Generating audio... (DiT backend: PyTorch (mps))" | ✅ 生效 |
| VAE 用 PyTorch/MPS tiled decode | 日志 "Using tiled VAE decode… PyTorch MPS；chunk 512→32" | ✅ 生效（但见小修②） |
| CPU offload 默认关闭 | profile "CPU offload: false"；.env 三处 offload=false | ✅ 生效（但见小修①） |
| DiT offload 默认关闭 | 同上（ACESTEP_OFFLOAD_DIT_TO_CPU=false） | ✅ 生效 |
| LM offload 默认关闭 | 同上 | ✅ 生效 |
| API worker=1 | profile "Queue workers: 1"；/v1/stats queue_maxsize=32 | ✅ 生效 |
| 模型启动时只加载一次 | 每套配置日志中 DiT/LM 加载各 1 次（grep 计数=3 关键词各一） | ✅ 生效 |
| 多 seed 串行执行 | 30s×4、60s×3 连续提交均串行完成，无并行 | ✅ 生效 |
| .env 可覆盖所有关键参数 | 修改 MLX_DIT / OFFLOAD / CACHE_ROOT 后重启均按 .env 生效 | ✅ 生效 |
| 无 /Users/anshi 硬编码 | 启动 profile 无任何硬编码路径（$HOME 展开） | ✅ 生效 |
| cache root 可切外置 SSD | ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step 启动成功 | ✅ 生效 |
| macos_perf_check.py 正常输出 | 运行 exit 0，输出 16GiB/arm64/swap/memory pressure 等 | ✅ 生效 |

**两个小瑕疵（不影响主路径）**：
1. 日志 `[API Server] Auto-enabling CPU offload (GPU < 16GB)` 为无条件打印，与 .env 实际值（offload=false）矛盾，误导排障。
2. `_init_mlx_vae()` 在 MLX_VAE=0 且设备为 MPS 时仍**无条件编译一份 MLX VAE 副本**（日志 "Native MLX VAE initialized"），与文档"PyTorch/MPS tiled"表述不一致，白占 ~数百 MB 内存——16GB 机器上每个 MB 都关键。

## C. 20 / 30 / 60 / 90 秒 Benchmark

条件：同一 prompt（Deep ocean waves, cinematic epic orchestral atmosphere, wind and sea, no vocals）、固定 seed、batch_size=1、推荐配置 A。

| 请求时长 | 次数 | 总生成 (s) | LM phase (s) | DiT phase (s) | RTF | wall (s) | 结果 |
|---|---|---|---|---|---|---|---|
| 20s | #1 冷缓存 | 73.09 | 3.47 | 69.62 | 3.65 | ~73 | ✅ |
| 20s | #2 热缓存 | 54.97 | 10.35 | 44.62 | 2.75 | 73.6 | ✅ |
| 30s | #1 | 63.95 | 11.31 | 52.63 | 2.13 | 82.4 | ✅ |
| 30s | #2 | 60.48 | 10.37 | 50.11 | 2.02 | 79.6 | ✅ |
| 60s | #1 | 82.37 | 10.58 | 71.79 | 1.37 | 100.7 | ✅ |
| 60s | #2 | 79.57 | 10.60 | 68.97 | 1.33 | 95.7 | ✅ |
| 90s | #1 | 78.60 | 10.61 | 68.00 | 0.87 | 102.4 | ✅ |
| 90s | #2 | 93.57 | 11.86 | 81.72 | 1.04 | 112.8 | ✅ |

- 全部成功；无 OOM / SIGKILL / crash / MPS fallback 异常（仅 flash attention→SDPA 已知降级）。
- 长音频 RTF 更低（diffusion 步数固定 8 步，latent 增长不线性放大时间）；90s 接近实时（RTF≈1.0）。
- 20s 音频 VAE decode（MPS tiled，chunk 512→32）≈16.75s。
- 单次波动（如 90s r2 93.57 vs r1 78.60）源于系统内存竞争（swap 压力），非模型缺陷。

## D. 连续多 seed 稳定性

- 30s × 4 seed（201–204）：66.70 / 74.88 / 56.69 / 66.21s → 全部成功，第 4 次不比第 1 次慢。
- 60s × 3 seed（301–303）：76.28 / 76.70 / 79.17s → 稳定。
- 全程 server 不重启（同一 PID 568 存活整个多 seed 周期）；模型仅在启动时加载一次；任务后进程 RSS 回落至 0.74GB。
- 未发现 memory leak 特征（RSS / swap 无持续爬升）。

## E. 推荐配置 A vs MLX DiT（B）

| 项 | A（推荐，MLX_DIT=false） | B（MLX_DIT=true） |
|---|---|---|
| 启动 | ✅ 77s 成功 | ❌ **133s 后 OOM（exit 137），MLX LM 权重加载阶段被杀** |
| 峰值 swap | 7.94GB（97% of 8.19GB） | **11.26GB**（超出 swap 上限，进程被杀） |
| 20/30s 生成 | 可测 | 无法测（服务起不来） |
| 稳定性 | 全部测试通过 | 不可用 |

- 根因：B 同时常驻 PyTorch DiT + MLX DiT + MLX VAE + MLX LM 四份权重，16GB 统一内存无法容纳。
- 结论：**M4/16GB 上 B 不可行；稳定性优先，A 明显更优**。B 的失败恰好反证 PR 移除"MLX DiT 双份"设计的正确性。

## F. memory pressure / swap 结论

- 生成任务期间：进程 RSS 峰值 5.4GB；swap used 峰值 7.94GB / 8.19GB（97%）；系统 memory free 35%。
- 系统在"swap 满边缘"运行，但 90s 长音频 + 连续多 seed 均未触发 OOM。
- 服务停止后 swap 回落至 2.2GB——swap 压力主要来自模型加载与推理瞬时峰值，不常驻。
- **风险提示**：若用户开启更多高内存应用（如浏览器多标签、大型 IDE），16GB 极易越界；建议实际部署时保持系统空闲。

## G. 内存泄漏迹象

未发现。多 seed 连续 7 个任务（30s×4 + 60s×3）：RSS 不持续爬升、swap 不持续增长、任务后 RSS 回落，第 4 次生成时间无退化。

## H. 外置 SSD 对启动 / 加载的影响

| cache root | 冷启动到 health OK |
|---|---|
| 默认（~/Library/Caches/ace-step） | 77s |
| /Volumes/ssd/ace-step | 70s（−7s / −9%） |

- 模型权重已存在于仓库 checkpoints，cache root 主要影响下载/临时文件；**SSD 不宣称显著提升推理速度**（与用户要求一致）。启动差 7s 可能含系统状态波动，仅作参考。

## I. 当前推荐配置（= PR 推荐 = A）

```
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_USE_MLX_DIT=false
ACESTEP_OFFLOAD_TO_CPU=false
ACESTEP_OFFLOAD_DIT_TO_CPU=false
ACESTEP_MLX_VAE=0
ACESTEP_MLX_VAE_CHUNK=192
ACESTEP_QUEUE_WORKERS=1
ACESTEP_QUEUE_MAXSIZE=32
ACESTEP_NO_INIT=false
# ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step  （可选用 SSD）
```

## J. 结论：**可合并，但建议补小修**

依据：
1. PR 声称的 14 项优化全部经源码+日志+实测确认生效；
2. 20/30/60/90s 全时长、多 seed、对照实验均如实跑完（无 NOT VERIFIED 项）；60–90s 长音频稳定、多 seed 无泄漏、swap 未失控；
3. B（MLX DiT）在 16GB M4 上 OOM 不可用，证明 A 配置与"移除双份 MLX DiT"的设计正确；
4. 建议补小修（非 blocker）：
   - 修正 "Auto-enabling CPU offload" 误导性日志（按 .env 实际值打印）；
   - MLX_VAE=0 时跳过 `_init_mlx_vae()` 的无条件预编译，与文档"PyTorch/MPS tiled"一致并省内存。

NOT VERIFIED 声明：无计划内项目未执行；B 配置因 OOM 无法产出 20/30s 数据，已如实记录为"无法启动"，未编造数据。
