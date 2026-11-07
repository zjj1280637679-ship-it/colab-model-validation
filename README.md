# Colab model validation · BF16-first

可复用的两个 Colab Notebook，附可追溯 JSON 证据及运行时释放步骤。不分发模型权重，不承诺免费 GPU。

## 本轮结论与边界

- **Qwen：主生成路径 BF16 通过，MTP 辅助分支未加载/未启用；未验证视觉输入和长上下文。** 三项文本生成完成：中文解释、17×23=391、`square(x)`。已加载 27,356,728,560 个 BF16 CUDA 参数；checkpoint 多出的 15 个 BF16 MTP 张量共 424,699,392 参数，header/meta key 审计无其他缺失和形状差异。不宣称全部权重加载或速度优化。
- **Qwen Image 2.1：一次 2048×2048 RGBA 文生图完成，40 步，83.44 秒。** 三个学习组件实际加载均为 BF16、cuda:0、无量化；上游 VAE 存储源为 FP32，本次加载转为 BF16，主 Transformer 与文本编码器上游为 BF16，峰值已分配显存 56.49 GiB。操作员查看了 Notebook 渲染，观察到红茶壶、绿植物、木桌和文字卡片。**图片编辑与透明生成未测**，RGBA 格式本身不证明透明生成通过。
- **MiniMax-H3：未运行。** 许可地域适用性未明确，闭源 Context-IR / Regenerate2K 不提供，不使用其他模型冒充。
- 两路独立图像基线验收通过；Qwen checkpoint/meta 审计已完成。这里的“通过”只限上述小范围功能验证，不是完整能力、正确率、安全性或性能基准。

**证据限制：** 原 PNG/ZIP 未成功导出，因此仓库没有原 PNG；JSON 中图片 SHA-256 是运行时报告值，未用导出字节独立验证。私密 UI 日志、账号截图和个人 Colab 链接不公开。详情见 [证据清单](evidence/manifest.json) 与 [来源说明](evidence/PROVENANCE.md)。

## 打开 Notebook

- [Qwen 文本 BF16](notebooks/qwen3_8_bf16.ipynb)
- [Qwen Image BF16](notebooks/qwen_image_2_1_bf16.ipynb)

下载对应 `.ipynb`，在 Colab 上传打开，选择 **A100 80GB**，按单元格顺序执行。也可使用 Colab 的 GitHub 打开方式定位本仓库。两个 Notebook 独立运行，不需要先跑文本再跑图像。

## 环境与精度

实测 Python 3.13.16、torch 2.14.1+cu130（Colab 提供）、Transformers 5.22.1、Accelerate 1.15.0、Hugging Face Hub 1.33.0、Safetensors 0.8.0。Diffusers 固定官方 commit `84532a058702c74ae5d1dd6ceb05683fc5d5e768`，报告版本 0.42.0.dev0。

`requirements.txt` 固定已观测直接依赖，不是完整传递依赖锁。保留 Colab 提供的 torch/CUDA，不盲目替换。环境更新后应重测。A100 40GB/T4、CPU offload、FP8/INT8/INT4/GGUF 均未在本轮验证。未做专用内核优化。

Notebook 不挂载 Drive、不读取 Hugging Face token、不建立公网隧道。预检先检查设备；模型 revision 固定。GPU/RAM/磁盘不足应停止，不自动降精度。

## 许可

运行前阅读 [第三方许可说明](THIRD_PARTY_NOTICES.md)。Qwen Image 为研究用途许可，不能默认商业使用。MiniMax-H3 有地域等限制，未确认资格不下载、不运行，不提供规避流程。仓库不分发权重。

## 随手关灯与证据导出

运行开始即设置 25 分钟 `runtime.unassign()` 辅助计时器；它不能保证应对解释器崩溃或阻塞。请主动监督下载与运行，必要时提前停止。证据保存在 `/content/a100_validation`；导出后执行关机单元格，并检查 **Manage sessions** 已无该运行时。清空 CUDA 缓存不等于结束计费。

本轮操作员确认 0 活动会话、0 CU/小时。余额 UI 差值 1.71 CU 仅是当时读数，可能结算延迟，不是后续运行报价。计时器触发可能丢失未导出文件；下载失败不要无限占用 GPU。

## 开发检查

```sh
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
```

这只检查打包/语法/证据结构，不执行模型。生成 Notebook 清空输出。脚本与 Notebook 在原始运行后做了清理和独立运行包装修改，未重新整套执行；查看来源说明。禁止提交 token、权重、浏览器状态、账号截图。
