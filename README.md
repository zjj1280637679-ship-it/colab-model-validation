# Colab model validation · BF16-first

可复用的三个 Colab Notebook，附可追溯 JSON 证据及运行时释放步骤。不分发模型权重，不承诺免费 GPU。

## 本轮结论与边界

- **Qwen：主生成路径 BF16 通过，MTP 辅助分支未加载/未启用；未验证视觉输入和长上下文。** 三项文本生成完成：中文解释、17×23=391、`square(x)`。已加载 27,356,728,560 个 BF16 CUDA 参数；checkpoint 多出的 15 个 BF16 MTP 张量共 424,699,392 参数，header/meta key 审计无其他缺失和形状差异。不宣称全部权重加载或速度优化。
- **Qwen Image 2.1：一次 2048×2048 RGBA 文生图完成，40 步，83.44 秒。** 三个学习组件实际加载均为 BF16、cuda:0、无量化；上游 VAE 存储源为 FP32，本次加载转为 BF16，主 Transformer 与文本编码器上游为 BF16，峰值已分配显存 56.49 GiB。操作员查看了 Notebook 渲染，观察到红茶壶、绿植物、木桌和文字卡片。**图片编辑与透明生成未测**，RGBA 格式本身不证明透明生成通过。
- **MiniMax-H3：成熟作者 Colab 工作流的高精度适配 v2 已完成原生20/20步生成和媒体结构检查。** 864×480、124帧、24fps、32kHz双声道；0:00/0:02/0:05视觉抽查通过，非逐帧QA。不是全组件BF16，计算dtype/真实峰值未测；原MP4未取回，独立听音与媒体hash未验证。节点地域仍未知，不作许可资格结论。见 [H3说明](h3_author/README.md)。
- 两路独立图像基线验收通过；Qwen checkpoint/meta 审计已完成。这里的“通过”只限上述小范围功能验证，不是完整能力、正确率、安全性或性能基准。

**证据限制：** 原 PNG/ZIP 未成功导出，因此仓库没有原 PNG；JSON 中图片 SHA-256 是运行时报告值，未用导出字节独立验证。私密 UI 日志、账号截图和个人 Colab 链接不公开。详情见 [证据清单](evidence/manifest.json) 与 [来源说明](evidence/PROVENANCE.md)。

## 打开 Notebook

- [Qwen 文本 BF16](notebooks/qwen3_8_bf16.ipynb) · [在 Colab 一键打开](https://colab.research.google.com/github/zjj1280637679-ship-it/colab-model-validation/blob/main/notebooks/qwen3_8_bf16.ipynb)
- [Qwen Image BF16](notebooks/qwen_image_2_1_bf16.ipynb) · [在 Colab 一键打开](https://colab.research.google.com/github/zjj1280637679-ship-it/colab-model-validation/blob/main/notebooks/qwen_image_2_1_bf16.ipynb)

下载对应 `.ipynb`，在 Colab 上传打开，选择 **A100 80GB**，按单元格顺序执行。也可使用 Colab 的 GitHub 打开方式定位本仓库。Qwen 两个 Notebook 独立运行；H3 使用作者原生 ComfyUI 脚本与工作流的 Notebook 封装，按其独立说明运行。

- [H3 作者工作流高精度封装](h3_author/minimax_h3_author_bf16.ipynb) · [在 Colab 一键打开](https://colab.research.google.com/github/zjj1280637679-ship-it/colab-model-validation/blob/main/h3_author/minimax_h3_author_bf16.ipynb)（默认运行与许可确认关闭）

## Qwen 环境与精度

实测 Python 3.13.16、torch 2.14.1+cu130（Colab 提供）、Transformers 5.22.1、Accelerate 1.15.0、Hugging Face Hub 1.33.0、Safetensors 0.8.0。Diffusers 固定官方 commit `84532a058702c74ae5d1dd6ceb05683fc5d5e768`，报告版本 0.42.0.dev0。

`requirements.txt` 固定已观测直接依赖，不是完整传递依赖锁。保留 Colab 提供的 torch/CUDA，不盲目替换。环境更新后应重测。A100 40GB/T4、CPU offload、FP8/INT8/INT4/GGUF 均未在本轮验证。未做专用内核优化。

Notebook 不挂载 Drive、不读取 Hugging Face token、不建立公网隧道。预检先检查设备；模型 revision 固定。GPU/RAM/磁盘不足应停止，不自动降精度。

## 许可

运行前阅读 [第三方许可说明](THIRD_PARTY_NOTICES.md)。Qwen Image 为研究用途许可，不能默认商业使用。MiniMax-H3 有地域等独立许可限制；默认运行与许可确认关闭，运行记录不构成对地域未知节点的法律资格证明，不提供规避流程。仓库不分发权重。

## 随手关灯与证据导出

Qwen 运行开始即设置单一 25 分钟 `runtime.unassign()` 生命周期计时器；重复启动单元格取消旧计时器并保留原截止时间，不延长预算；显式关机使用同一生命周期状态。它不能保证应对解释器崩溃或阻塞。请主动监督下载与运行，必要时提前停止。每次预检建立 `/content/a100_validation/<模型>-<随机ID>` 唯一目录；导出仅包含该模型明确白名单内文件，禁止符号链接，ZIP 中除 manifest 自身外的每个条目均与 SHA-256 manifest 一致；导出后执行关机单元格，并检查 **Manage sessions** 已无该运行时。清空 CUDA 缓存不等于结束计费。

Qwen 本轮操作员确认 0 活动会话、0 CU/小时。余额 UI 差值 1.71 CU 仅是当时读数，可能结算延迟，不是后续运行报价。计时器触发可能丢失未导出文件；下载失败不要无限占用 GPU。

## 开发检查

```sh
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
```

这只检查打包/语法/证据结构，不执行模型。生成 Notebook 清空输出。Qwen 脚本与 Notebook 在原始运行后做了清理和独立运行包装修改，未重新整套执行；查看来源说明。H3 活动执行程序与实际 v2 完全一致，发布仅更新说明和 Notebook 文字。禁止提交 token、权重、浏览器状态、账号截图。

## 预检与导出保护

文本采用可用 RAM ≥64 GiB、磁盘 ≥80 GiB；图像采用 RAM ≥48 GiB、磁盘 ≥55 GiB。这是本轮后加的保守保护门槛，不是实测最低要求。约55.6 GB/33.1 GB下载量分别约51.8 GiB/30.8 GiB（1 GiB=2³⁰字节），另预留加载、缓存与导出空间。本轮预检可用 RAM 164.07 GiB、磁盘235.58 GiB。固定 revision 元数据与实际模型加载一致，不用浮动 HEAD 替代。

依赖来源由已安装包的 direct_url.json 读取；来源未知为 null，未安装也如实记录，不将计划的 Diffusers commit 冒充文本运行中实际安装来源。

H3 许可 II/V.4 涉及使用、复制等发生地域。操作者位于适用地域不意味着地域未知的托管节点已获许可。官方 FAQ 的全球 API 指 MiniMax 自营 API，不覆盖 Colab 自托管权重；[相关官方社区提问](https://huggingface.co/MiniMaxAI/MiniMax-H3/discussions/12) 未视为许可授权。工程不分发 H3 权重或原媒体；现包含默认关闭的成熟作者工作流适配及脱敏摘要，不以实测替代许可适用性审查。

图像单元格若检测到 Diffusers 已导入会停止，要求重启运行时并从开头执行，防止新安装 metadata 与旧内存模块混用。未来结果分别记录实际安装来源/commit 与预期 commit；历史 JSON 保持原样，其 commit 字段来自当时安装 pin，不伪称新增来源检查已在原运行执行。

## H3 作者工作流 v2

[实现、MIT归属、固定源与完整边界](h3_author/README.md)。原作者提供脚本与工作流，本项目封装ipynb。总计26分03.625秒，图执行333.33秒/轮询334.309秒；DiT与H3专用conditioner为BF16加必要FP32，视频VAE FP16、音频VAE FP32，仅测原生loader返回时CPU参数存储。不是全组件BF16或计算dtype证明。GPU已释放，GUI观察RAM114.8GB/VRAM77.9GB及3.44CU余额差均非真实峰值/最终结算。旧自造Diffusers H3不计入通过结果，未纳入本次发布。
