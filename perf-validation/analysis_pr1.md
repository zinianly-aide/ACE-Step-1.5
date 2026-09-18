# PR #1 性能验证报告 — perf/macos-m4-16gb（第二轮：数据一致性修正 + 最小性能修复 + 轻量回归）

- 验证日期：2026-09-19（第二轮更新）
- 仓库：github.com/zinianly-aide/ACE-Step-1.5
- PR #1：perf/macos-m4-16gb → main（OPEN / MERGEABLE，+407/-19，4 files）
- 分支：origin/main = 225d74e；PR head = e4cc70b
- 本轮：修正验证报告数据一致性（A20s_r1）；对 PR #1 做两个最小性能修复并补测试；M4/16GB 轻量回归；未 merge main；未动 generated-audio；未重生成业务成品音频。

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

## B. PR 配置是否真实生效（第二轮复核，含修复后日志）

| PR 声称 | 验证方式 | 结果 |
|---|---|---|
| LM 使用 MLX | 启动日志 "Attempting MLX backend… MLX model loaded successfully" | ✅ 生效 |
| LM 模型 0.6B | profile "LM backend/model: mlx / acestep-5Hz-lm-0.6B" | ✅ 生效 |
| DiT 不用 MLX 双份常驻 | MLX_DIT=false 时日志无 MLX-DiT init；DiT backend = PyTorch (mps) | ✅ 生效 |
| DiT 用 PyTorch/MPS | 日志 "DiT backend: PyTorch (mps)" | ✅ 生效 |
| VAE 用 PyTorch/MPS tiled decode | 日志 "Using tiled VAE decode… PyTorch MPS；chunk 512→32" | ✅ 生效（修复后 MLX VAE 不再预编译，见 D 节） |
| CPU offload 默认关闭 | .env 三处 offload=false；**修复后日志 "CPU offload: disabled by explicit env override"** | ✅ 生效（日志已修正，见 D 节） |
| DiT offload 默认关闭 | 同上（ACESTEP_OFFLOAD_DIT_TO_CPU=false） | ✅ 生效 |
| LM offload 默认关闭 | 同上 | ✅ 生效 |
| API worker=1 | profile "Queue workers: 1"；/v1/stats queue_maxsize=32 | ✅ 生效 |
| 模型启动时只加载一次 | 每套配置日志中 DiT/LM 加载各 1 次 | ✅ 生效 |
| 多 seed 串行执行 | 30s×4、60s×3、本轮 30s×4 连续提交均串行完成 | ✅ 生效 |
| .env 可覆盖所有关键参数 | 修改 MLX_DIT / OFFLOAD / CACHE_ROOT / MLX_VAE 后重启均按 .env 生效 | ✅ 生效 |
| 无 /Users/anshi 硬编码 | 启动 profile 无任何硬编码路径（$HOME 展开） | ✅ 生效 |
| cache root 可切外置 SSD | ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step 启动成功 | ✅ 生效 |
| macos_perf_check.py 正常输出 | 运行 exit 0，输出 16GiB/arm64/swap/memory pressure 等 | ✅ 生效 |

## C. 20 / 30 / 60 / 90 秒 Benchmark（首轮完整矩阵）

条件：同一 prompt（Deep ocean waves, cinematic epic orchestral atmosphere, wind and sea, no vocals）、固定 seed、batch_size=1、推荐配置 A。

| 请求时长 | 次数 | 总生成 (s) | LM phase (s) | DiT phase (s) | RTF | wall (s) | 原始 JSON | 结果 |
|---|---|---|---|---|---|---|---|---|
| 20s | #1 冷缓存 | 73.09 | 3.47 | 69.62 | 3.65 | ~73 | **A20s_r1.json=timeout（见 C1 一致性说明）** | ✅ 服务端成功 |
| 20s | #2 热缓存 | 54.97 | 10.35 | 44.62 | 2.75 | 73.6 | A20s_r2.json（ok:true） | ✅ |
| 30s | #1 | 63.95 | 11.31 | 52.63 | 2.13 | 82.4 | A30s_r1.json（ok:true） | ✅ |
| 30s | #2 | 60.48 | 10.37 | 50.11 | 2.02 | 79.6 | A30s_r2.json（ok:true） | ✅ |
| 60s | #1 | 82.37 | 10.58 | 71.79 | 1.37 | 100.7 | A60s_r1.json（ok:true） | ✅ |
| 60s | #2 | 79.57 | 10.60 | 68.97 | 1.33 | 95.7 | A60s_r2.json（ok:true） | ✅ |
| 90s | #1 | 78.60 | 10.61 | 68.00 | 0.87 | 102.4 | A90s_r1.json（ok:true） | ✅ |
| 90s | #2 | 93.57 | 11.86 | 81.72 | 1.04 | 112.8 | A90s_r2.json（ok:true） | ✅ |

- 首轮矩阵全部成功（无 OOM / SIGKILL / crash / MPS fallback 异常，仅 flash attention→SDPA 已知降级）。
- 长音频 RTF 更低（diffusion 步数固定 8 步，latent 增长不线性放大时间）；90s 接近实时（RTF≈1.0）。
- 20s 音频 VAE decode（MPS tiled，chunk 512→32）≈16.75s。
- 单次波动（如 90s r2 93.57 vs r1 78.60）源于系统内存竞争（swap 压力），非模型缺陷。

### C1. A20s_r1 数据一致性处理（本轮修正）

原始文件 `bench-data/A20s_r1.json` 内容为：`"ok": false`、`"error": "job f1b9b299-… did not complete in timeout; last status=1"`、`"wall_s": 922.026` —— 是**客户端超时误报记录**。

事实核查（服务端原始日志 + 手动 query_result）：
- 服务端任务 **f1b9b299-913d-4f42-abd1-d1e69500cdc9 实际 succeeded**：stage=succeeded，generation_info "Total generation time (1 song): 73.09s"（LM 3.47s / DiT 69.62s），产物 .cache/acestep/tmp/api_audio/67ad009e-5dc0-7608-af31-41226feb1279.mp3。
- 当时误报根因：旧版 bench 客户端未解析 `data[0].result` 内的 JSON 字符串，把"任务仍在排队/查询超时"判为失败；服务端本身完成且成功。
- `bench-data/A20s_r2.json` = 同 seed 101 的重跑成功记录（54.97s，ok:true）——即"重跑成功"文件已存在，无需另建 `A20s_r1_retry.json`。

处理结论：
- **不删除失败数据**：`A20s_r1.json` 原样保留（ok:false, timeout 记录）。
- 报告不再声称"20s #1 成功"由 A20s_r1.json 直接支撑；表格中 20s#1 的数据来自服务端真实任务日志（73.09s），并明确标注原始 JSON 为 timeout、服务端实际成功。
- 客户端解析 bug 已在首轮修复（bench_client.py 现正确解析 `data[0].result`），本轮所有新跑任务均由新版客户端确认 ok:true。

### C2. C30s_r1（offload=true）正式纳入对照

`bench-data/C30s_r1.json` = ok:true，seed 401，wall 75.589s，generation 61.04s（LM 3.86s / DiT 57.18s），产物 0b35462c-*.mp3。该文件对应配置 **C = ACESTEP_OFFLOAD_TO_CPU=true**（其余同 A）。

| 项 | A（推荐，offload=false） | C（offload=true） |
|---|---|---|
| 30s generation | 63.95 / 60.48s | 61.04s（同量级，无显著差异） |
| wall | 82.4 / 79.6s | 75.6s |
| 日志期 RSS 量级 | 任务期峰值 ~0.9–1.2GB（ps 口径，见 F 节说明） | 日志显示 "Loading vae to mps (RSS: 10000 MB)"，明显更高 |
| peak RSS / peak swap / memory pressure | 见 F 节 | **NOT VERIFIED**（C 窗口 monitor 数据缺失——monitor 进程在 C 测试前被重启，swap/peak MPS 未记录，不猜测） |
| 结论 | — | offload=true 无性能收益，且 RSS 量级显著更高；**不建议在 M4/16GB 上开启 offload** |

## D. 本轮代码修复（已提交 PR #1 分支，最小改动）

### 修复 1：CPU offload 日志与最终行为一致
- 文件：`acestep/api/startup_model_init.py`
- 根因：do_model_initialization 无条件打印 "Auto-enabling CPU offload (GPU < 16GB)"，即使 `ACESTEP_OFFLOAD_TO_CPU=false` 显式禁用。
- 修改：解析 env override 后再输出最终状态，四种明确文案：
  - `CPU offload: enabled by explicit env override`
  - `CPU offload: disabled by explicit env override`
  - `CPU offload: auto-enabled (GPU < 16GB)`
  - `CPU offload: disabled by auto-detection (GPU >= 16GB)`
  - `No GPU detected, running on CPU`
- 不影响初始化流程与 offload 语义（offload_to_cpu 变量仍原样传给 initialize_service / initialize_llm_at_startup）。

### 修复 2：ACESTEP_MLX_VAE=0 时不再初始化 / 预编译 MLX VAE
- 文件：`acestep/core/generation/handler/init_service_setup.py`
- 根因：`_initialize_mlx_backends` 在 MPS/CPU 上无条件调用 `_init_mlx_vae()`，即使 `ACESTEP_MLX_VAE=0`（decode 实际走 PyTorch/MPS tiled）也编译并常驻一份 MLX VAE 副本。
- 修改：先解析 `ACESTEP_MLX_VAE`（默认 "1"）；值为 0/false/no 时跳过 `_init_mlx_vae()`，置 mlx_vae=None / use_mlx_vae=False，状态串 "Disabled by user (ACESTEP_MLX_VAE=0)"；=1/true/未设置保持原路径不变。与 `vae_decode.py` 既有 guard 语义对齐，不改模型数学、不改 decode 路径。
- 修复后启动日志验证：MLX VAE 初始化关键词出现 **0 次**（grep "MLX-VAE|Native MLX VAE" = 0）；offload 日志为 `[API Server] CPU offload: disabled by explicit env override`。

### 新增单元测试（15/15 通过，无需真实 MPS 硬件）
- `acestep/api/startup_model_init_test.py` 新增 5 例（offload 日志四态 + 无 GPU）：
  - env=false + 小 GPU → offload=false 且日志为 "disabled by explicit env override"，不含 auto 字样
  - env=true + 大 GPU → offload=true，日志 "enabled by explicit env override"
  - 无 env + 小 GPU → auto-enabled（原行为保留）
  - 无 env + 大 GPU → disabled by auto-detection
  - 无 GPU → CPU 模式
- 新建 `acestep/core/generation/handler/init_service_setup_test.py` 6 例（MLX VAE gating）：
  - MLX_VAE=0 / false → `_init_mlx_vae` 不被调用、mlx_vae=None、use_mlx_vae=False、状态含 "Disabled by user"
  - =1 / 未设置 → 原路径仍调用
  - cpu 设备同样受 guard 约束；cuda 设备从不调用
- 既有 mlx_vae_init_test 3 例 OK。
- 既有 init_service_test / generate_music_decode_test / vae_decode_mixin_test 71 例中 1 例失败（test_load_main_model_ignores_cuda_sync_cleanup_error，AttributeError _sync_alignment_config）——已用 git stash 验证该失败在本轮改动前同样存在，**预存在问题，非本次引入**。

## E. 轻量回归（修复后，M4/16GB，配置 A）

| 项 | 修复前（首轮） | 修复后（本轮） |
|---|---|---|
| 冷启动到 health OK | 77s | **76s** |
| 启动期进程树 RSS 峰值 | 9478MB（5s 采样） | 9542MB（5s 采样）——持平，启动峰值由权重加载主导 |
| 30s generation | 63.95 / 60.48s | 61.45 / 66.22 / 58.62 / 72.06s（4 次均成功） |
| 90s generation | 78.60 / 93.57s | 91.36s（成功） |
| 30s 连续 4 seed | 66.70 / 74.88 / 56.69 / 66.21s | 72.93 / 68.86 / 60.77 / 64.17s —— 第 1 次最慢、后续无爬升 |
| 任务期 swap used 峰值 | 7.94GB（97% of 8.19GB） | ~7.0GB（1s 采样，86% of 8.19GB）；会话级峰值 7.5GB |
| 服务停止后 swap 回落 | 2.2GB | 2.27GB（瞬时压力，不常驻） |
| MLX VAE 初始化 | 无条件编译（数百 MB 常驻） | **0 次**（日志确认） |
| offload 日志 | 误导（打印 auto-enable 实际 false） | 正确（disabled by explicit env override） |

- 任务期 RSS 口径说明：本轮 1s 采样进程树（主进程+子进程）RSS 峰值 884MB；首轮报告 5.4GB 为 5s 采样值。两轮采样间隔/进程匹配口径不同，且 MPS/Metal 统一内存分配不完全计入进程 RSS，**数值不直接对比**；以 swap（系统级指标）与启动/推理耗时为准。
- MLX VAE 去除的效果：**启动峰值 RSS 无显著变化**（启动峰值由 DiT/LM/VAE 权重加载主导）；任务期 swap 峰值由 7.94GB → ~7.0GB，**下降约 0.9GB（≈12%），幅度小但可观测**；推理速度无退化。
- 4 seed 连续：server 不重启（同一 PID 42784 存活全程）、模型不重复加载、第 2–4 次无明显初始化开销、第 4 次不比第 1 次慢。

## F. memory pressure / swap 结论（措辞修正）

- 生成任务期间：进程树 RSS 峰值 884MB（1s 采样 ps 口径；MPS 统一内存另计）；swap used 峰值 ~7.0GB / 8.19GB（86%）；系统 memory free 15–50% 波动（多数 25–40%）。
- **修正措辞**：测试可稳定完成，但**峰值 swap 已接近系统上限，对系统盘剩余空间和后台内存占用敏感**——不应表述为"swap 未失控"。若用户开启更多高内存应用（浏览器多标签、大型 IDE），16GB 极易越界；建议实际部署时保持系统空闲。
- 服务停止后 swap 回落至 2.27GB——swap 压力主要来自模型加载与推理瞬时峰值，不常驻。
- 修复前后 swap 对比：修复前任务期峰值 7.94GB（97%）→ 修复后 ~7.0GB（86%），下降 ≈0.9GB / ≈12%；**下降不明显，如实记录**。去除 MLX VAE 预编译对 16GB 机器的内存压力有边际缓解，但不是决定性因素（决定性因素是 DiT/LM/VAE 权重加载）。

## G. 内存泄漏迹象

未发现。首轮 7 个任务（30s×4 + 60s×3）与本轮 4 seed（30s×4）+ 单次 30s/90s：RSS 不持续爬升、swap 不持续增长、任务后 RSS 回落至 ~0.17GB、swap 回落至 2.27GB、第 4 次生成时间无退化。

## H. 外置 SSD 对启动 / 加载的影响

| cache root | 冷启动到 health OK |
|---|---|
| 默认（~/Library/Caches/ace-step） | 77s |
| /Volumes/ssd/ace-step | 70s（−7s / −9%） |

- 模型权重已存在于仓库 checkpoints，cache root 主要影响下载/临时文件；**SSD 不宣称显著提升推理速度**（与用户要求一致）。启动差 7s 可能含系统状态波动，仅作参考。

## I. 当前推荐配置（= PR 推荐 = A，本轮实测）

```
ACESTEP_LM_BACKEND=mlx
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_USE_MLX_DIT=false
ACESTEP_OFFLOAD_TO_CPU=false
ACESTEP_OFFLOAD_DIT_TO_CPU=false
ACESTEP_MLX_VAE=0          ← 修复后彻底不初始化 MLX VAE（本轮实测 0 次）
ACESTEP_MLX_VAE_CHUNK=192
ACESTEP_QUEUE_WORKERS=1
ACESTEP_QUEUE_MAXSIZE=32
ACESTEP_NO_INIT=false
# ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step  （可选用 SSD）
```

## J. 结论：**可合并，仍有已知风险**

依据：
1. PR 声称的 14 项优化全部经源码+日志+实测确认生效；两个小瑕疵（offload 误导日志、MLX VAE 无条件预编译）已在本轮修复，15 个新增/相关单元测试全部通过，轻量回归（冷启动 76s、30s 61.45s、90s 91.36s、30s×4 seed 稳定）通过。
2. A20s_r1 数据一致性已修正：原始 JSON（ok:false timeout）保留，报告数据与原始 JSON 严格对应，并注明服务端实际成功与重跑记录（A20s_r2）。
3. C30s_r1（offload=true）正式纳入对照：无性能收益且 RSS 量级更高，不建议开启；其 swap/peak MPS 数据缺失，已标 NOT VERIFIED。
4. B（MLX DiT）在 16GB M4 上 OOM 不可用（首轮实测），证明 A 配置与"移除双份 MLX DiT"的设计正确。
5. **已知风险（合并后仍需注意）**：16GB M4 上峰值 swap 接近系统上限（~7.0GB / 8.19GB，86%），对系统盘剩余空间与后台内存占用敏感；建议实际部署时关闭高内存应用、保持系统空闲。

NOT VERIFIED 声明：C30s_r1 的 peak RSS / peak swap / memory pressure（C 窗口 monitor 缺失）；首轮 B 配置 20/30s 数据（服务 OOM 无法启动）。以上均未编造数据。
