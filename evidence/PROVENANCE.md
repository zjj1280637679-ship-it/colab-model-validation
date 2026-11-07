# Evidence provenance and limitations

The three model/audit JSON files were parsed from real Colab UI output by the operator. Private UI transcripts are excluded. They are observed-output evidence, not independently downloaded original report files.

- `qwen3_8_bf16.json`: three text generations, BF16 parameters on CUDA, no quantization. The first explanation's free-compute wording is raw model output, not a guarantee of free Colab GPUs/TPUs.
- `qwen_checkpoint_audit.json`: all shard headers and meta-model keys/shapes checked. The only checkpoint-only tensors are 15 MTP tensors totaling 424,699,392 BF16 parameters. No model-only keys or shape mismatches. This is not inference with MTP enabled.
- `qwen_mtp_header_review.json`: independently inspected MTP shard-header entries.
- `qwen_image_2_1_bf16.json`: one 2048-square image, RGBA, 40 steps. Image editing and transparent-output generation were not tested.

The original image and ZIP were not successfully exported. The PNG hash is reported by runtime only, not independently verified against recovered bytes. The operator viewed the notebook rendering and observed the red teapot, green plant, wood table and labeled card. No original PNG or private screenshot is redistributed.

The executed Qwen script initially retained loop variable `p`, leaving 2.376 GiB allocated. The packaged script deletes it. The operator subsequently observed 0.008 GiB before the image run. The exporter excludes its own SHA manifest to avoid self-reference on repeated exports. Notebook installation, standalone timer, cleanup and exporter packaging changes have static checks but have not been rerun end-to-end as the final notebooks.

Observed direct versions are recorded; a complete transitive lock was not exported. Two independent image baseline reviews passed according to reviewer reports relayed by the operator. Source VAE weights are FP32 and were loaded as BF16; transformer and text encoder source weights are BF16. This is not a claim that every upstream file is BF16. No inference ran in repository CI or local packaging tests.
