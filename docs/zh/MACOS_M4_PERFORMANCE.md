# macOS / Apple Silicon 性能优化（M4 16GB）

本页针对 Mac mini M4 / 16GB 统一内存，目标是稳定生成 30–90 秒音乐并连续跑多个 seed，而不是追求“所有模块都使用 MLX”。

## 推荐架构

- 5Hz LM：MLX，使用 `acestep-5Hz-lm-0.6B`
- DiT：PyTorch / MPS
- VAE：PyTorch / MPS tiled decode
- API worker：1
- generation batch：1，多个 seed 串行提交

这样可以避免 MLX DiT 与 PyTorch DiT 同时常驻造成重复内存占用，并降低 60–90 秒长音频在 VAE decode 阶段的峰值压力。

## 快速启动

```bash
cp .env.macos.example .env
./run_api_macos.sh
```

如果模型缓存放在 Thunderbolt / NVMe 外置 SSD，可在 `.env` 设置：

```bash
ACESTEP_CACHE_ROOT=/Volumes/ssd/ace-step
```

注意：外置 SSD 主要改善下载、首次加载和磁盘占用；模型进入统一内存后，生成速度主要由 MPS/GPU 与内存带宽决定。

## 16GB 建议配置

```bash
ACESTEP_LM_BACKEND=mlx
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ACESTEP_USE_MLX_DIT=false
ACESTEP_MLX_VAE=0
ACESTEP_MLX_VAE_CHUNK=192
ACESTEP_OFFLOAD_TO_CPU=false
ACESTEP_OFFLOAD_DIT_TO_CPU=false
ACESTEP_QUEUE_WORKERS=1
ACESTEP_NO_INIT=false
```

`run_api_macos.sh` 已提供以上默认值，但 `.env` 可以覆盖它们。

## 为什么不建议全 MLX

16GB Apple Silicon 使用统一内存。LM、DiT、VAE、Python、MPS 与系统应用共享同一内存池。若同时保留 MLX DiT 和 PyTorch/MPS DiT，单次推理可能更快，但峰值内存更容易触发 swap，连续生成反而变慢或失败。

因此 M4 16GB 优先保证：

1. 单份 DiT 常驻；
2. 0.6B LM 使用 MLX；
3. 长音频使用 tiled MPS VAE；
4. 一个 worker 串行执行。

## 推荐生成工作流

先做 Preview，再做 Final：

- Preview：20–30 秒、batch=1、每幕 3–4 个 seed；
- 选定风格和 seed 后，再生成正式 30/60/90 秒版本；
- 不建议为了调 Prompt 每次都重新生成 90 秒；
- 不建议一次生成完整 180 秒再后期切分。

对于《芯舰破浪 · 扬帆远航》这类项目，推荐保持 90s / 30s / 60s 三幕生成，再用 FFmpeg 精确卡点。

## 运行前检查

```bash
.venv/bin/python scripts/macos_perf_check.py
```

脚本会检查：

- Apple Silicon 架构；
- 总统一内存；
- swap；
- memory pressure；
- ACE-Step 关键环境变量；
- Ollama / Docker / Unity / Android Studio 等潜在内存竞争进程；
- 本地 API 是否已启动。

长音频批量生成前，建议关闭不必要的 Docker、Ollama、Unity、Android Studio、浏览器大标签页以及其他本地 LLM。

## 并发策略

不要在 16GB 机器上同时启动多个生成进程，也不要把 API worker 提高到 2 或更多。

推荐：

```text
server 启动并只加载一次模型
  -> seed 1
  -> seed 2
  -> seed 3
  -> seed 4
  -> 再进入下一幕
```

如果客户端支持 `batch_size`，长音频仍建议保持 `batch_size=1`。

## 判断是否需要进一步优化

优先观察 Activity Monitor 的 Memory Pressure 和 Swap Used，而不是只看 GPU 利用率。

如果 Memory Pressure 持续绿色且 swap 基本不增长，再考虑尝试更激进的后端组合；如果 swap 快速增长，应优先减少并发、缩短 preview 时长或关闭其他模型服务。
