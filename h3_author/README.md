# H3：成熟作者 Colab 工作流的高精度适配 v2

[在 Colab 打开](https://colab.research.google.com/github/zjj1280637679-ship-it/colab-model-validation/blob/main/h3_author/minimax_h3_author_bf16.ipynb) · [原作者代码](https://github.com/shinshin86/comfy-agent/tree/a386b5cfaca565cc24cc592a0b480cb6c06971c9/scripts/colab/minimax_h3)

原作者提供 Colab 脚本和工作流，本项目将该作者程序封装为可复用 `.ipynb`，不是原作者已发布的成品 Notebook。

**实测状态：原生 ComfyUI 20/20 步生成及媒体结构检查通过；不是完整视听质量验收。** 这不是此前自造 Diffusers H3 流程。保留成熟作者 14 节点、17 连接的原生图、20 步、seed 42、864×480、124 帧、24 fps、原提示词和原生音频。

## 实际结果与精度边界

- NVIDIA A100-SXM4-80GB；总计 1563.625 秒（26 分 03.625 秒），图执行日志 333.33 秒，外层轮询 334.309 秒。不是专门性能基准。
- 解码检查：864×480、124 帧、24 fps、32000 Hz、双声道；非有限值检查与音频非静音数值检查通过。数值非静音不等于听感、语义或音画同步通过。
- **仅测原生 loader 返回时的 CPU 参数存储 dtype：** DiT 为 BF16 + 17,222,144 个 FP32 元素；H3 专用 conditioner 为 BF16 + 1 个 FP32 元素；视频 VAE FP16；音频 VAE FP32。没有报告量化参数/模块指示器。不是全组件 BF16，不证明计算/autocast dtype，也不是完整 checkpoint key/value 审计。
- GUI 最大观察 RAM 114.8 GB、VRAM 77.9 GB，只是观察值，不能称真实峰值。真实显存峰值未测。
- 原 ZIP/MP4 未取回，媒体 SHA-256 仅由运行时报告，未独立复算。0:00、0:02、0:05 三时间点视觉抽查通过：蓝调湖面、远山、灯笼清楚，灯笼缓移且略变大，水波倒影变化；不是逐帧 QA。独立听音仍未通过。
- 本轮运行时已释放。余额差 3.44 CU 不是最终结算或未来报价。

证据：[GPU 白名单摘要](../evidence/h3_v2_gpu_summary.json)、[CPU 白名单摘要](../evidence/h3_v2_cpu_summary.json)、[程序同一性与实际配置](../evidence/h3_v2_source_and_settings.json)。不发布原完整日志、账号、私有笔记/代理 URL 或权重。

## 怎么运行

1. 打开 Notebook，默认 `MODE="CPU_NO_WEIGHTS"`，`ALLOW_THIS_RUN=False`，许可确认关闭。
2. 先阅读模型许可与下面的适用性警告。配置、setup、startup、shutdown 四个代码单元按顺序执行。不要将运行开关当成法律资格判定。
3. CPU 模式只安装/启动原生 ComfyUI 并通过真实 `/object_info` 检查图结构；跳过全部模型下载、加载、采样及媒体验证。不要在 CPU 窗口手动排队模型图。
4. GPU 模式要求 A100 80GB、可用 RAM 至少 140 GiB、磁盘至少 123,599,845,928 字节加 20 GiB。门槛是保守保护，不保证不 OOM。四个权重下载约 123.6 GB（约 115.1 GiB）。
5. 单次生命周期从 setup 开始：CPU 12 分钟，GPU 整数 1–50 分钟、默认 50；第二单元不延长。停止/失败后确认 Manage sessions 无活动运行时。进程计时器不能保证内核崩溃后的释放。

实际 GPU 设置与公开默认设置分开保存于证据中；公开默认从未替用户启用。Notebook 的可执行程序、配置默认、精度 hook 与图保持冻结 v2；只更新说明文本。CPU/GPU 保存版两个执行单元完全相同，Colab 对 cell 1 的行尾空白做了归一化，证据中分别记录 raw SHA 与冻结 SHA。

## 来源、依赖与权重固定

- 原作者：Yuki Shindo / `shinshin86/comfy-agent`，固定 `a386b5cfaca565cc24cc592a0b480cb6c06971c9`。原文件保留在 `upstream/`，这些是历史原件，不是本适配的运行说明。
- ComfyUI 固定 `e01fb4c56b7a88149d469b99cbbfe3223d715054`；安装它的固定 requirements，不添加自定义节点。主要依赖固定为 frontend 1.48.7、workflow templates 0.11.41、embedded docs 0.5.9、comfy-kitchen 0.2.31、comfy-aimdo 0.4.13；另加 Jedi 0.19.2。保留 Colab 已安装 torch/torchvision/torchaudio/numpy/Pillow，并执行安装前后 pip 检查。**不是完整传递依赖锁**，CPU 与 GPU 实际版本分别记在摘要中。
- 权重仅来自 `Comfy-Org/MiniMax-H3@4cc1d817b6184899b41293954329f576cb5ae86b`。四个文件的路径、预期字节数及 SHA-256 见摘要 `planned_model_assets` 和 `01_setup.py`，下载后核对大小和 SHA。DiT BF16、conditioner BF16、视频 VAE FP16、音频 VAE FP32。
- `SHA256.json` 记录发布程序/Notebook/图；`build.py` 只生成本地文件，不下载模型、不运行 GPU。

## 归属与独立许可

作者代码 MIT ©2026 Yuki Shindo。完整 MIT 许可保留在 [LICENSE](LICENSE) 与 [upstream/LICENSE](upstream/LICENSE)。适配改动见 [CHANGES.md](CHANGES.md)，不能将适配误称原作者已验证的高精度组合。

ComfyUI 的软件许可及其依赖各自适用；本仓库不分发 ComfyUI 源树或权重。[MiniMax-H3 模型许可](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)独立于代码 MIT，涉及模型、输出及地域等限制，不因代码开源而豁免。Applicable Territory 排除美国、欧盟、英国、韩国。实际测试节点地域仍未知；已运行不等于获得法律适用性证明。`none_identified` 只表示该次审查未识别明确适用阻断事实，不是授权或无地域限制。对新部署必须自行审核；开关不能授予许可。

官方 FAQ 的全球 API 描述适用于 MiniMax 自营 API，不覆盖任意 Colab 自托管权重。本适配只覆盖公开 H3-Base 路径，不包含闭源 Context-IR / Regenerate2K。

## 已知运行历史

v1 在约 22 分钟完成依赖/下载与四组件存储审计后，于 0/20 初始化阶段被自设 25 分钟总截止终止。它是预算不足的未完成运行，不是 OOM 或兼容性失败。v2 仅调整有界预算与日志后完成本轮实测。

对外摘要仍不等于媒体全面质量通过。后续任何实质代码/依赖变化需要另行验证；本次发布没有新开 GPU。
