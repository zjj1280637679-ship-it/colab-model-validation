# Adaptation changes and tested-source identity

Base: shinshin86/comfy-agent@a386b5cfaca565cc24cc592a0b480cb6c06971c9, MIT ©2026 Yuki Shindo. Full original notices and selected source files remain in upstream/.

1. Replace author's INT8 diffusion and NVFP4 conditioner assets with fixed BF16 assets. Preserve author's FP16 video VAE and FP32 audio VAE. The author's original A100 verification was quantized, not this high-precision adaptation.
2. Preserve all 14 native workflow nodes, 17 connections, original prompt, 20 steps, seed 42, dimensions, frames, fps, audio and SaveVideo prefix. No alternate Diffusers implementation.
3. Remove cloudflared/public tunnel and Drive mounting from the active setup. Use loopback and Colab's authenticated window. No custom nodes, wildcard CORS, origin bypass, credential reading or keep-alive code added.
4. Protect the current Colab torch/torchvision/torchaudio/numpy/Pillow stack through constraints; add Jedi 0.19.2; perform pip checks. This is not a fully locked transitive environment.
5. Add default-off explicit run/license review controls, hardware/resource guards, unique run directory and bounded lifecycle shutdown. CPU mode skips every weight download and model execution, while inspecting real native node schemas.
6. Add read-only parameter-storage dtype/device/quantization summaries at exact native loader return points. No tensor movement or precision casting in the hooks. These summaries do not measure compute dtype, real memory peaks or full checkpoint correspondence.
7. Submit the author's native graph in GPU mode; numerically check decoded media and request a local whitelist ZIP. Runtime media hashes do not prove independently downloaded bytes or perceptual quality.
8. Version 2 changes only budget/log handling relative to v1: GPU maximum/default 50 minutes, CPU 12 minutes, single unextended lifecycle, quiet wget plus separate download/hash timings. The graph, inference cell and precision hook remained unchanged. v1_to_v2.diff and setup.diff retain mechanical differences.

Public packaging changes only the notebook documentation and adds summaries/tests; active program cells, config defaults, workflow, hook and setup/startup sources remain byte-identical to frozen v2. Original CPU and GPU saved cells are identical to each other; Colab stripped cell 1 trailing whitespace relative to the frozen source. The program-source digest excludes configuration and documentation; separate file SHA values record them. Evidence records the actual settings independently of the default-off distribution.
