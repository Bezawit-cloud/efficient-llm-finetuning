# Configuration Documentation

## Batch Size / Memory Note (Aug 2026)
The gpu_smoke_test.py originally used a hardcoded micro-batch of 4 with short sequences, which passed at 1.62GB peak VRAM and gave false confidence. The real training config uses per_device_train_batch_size=8 with gradient_accumulation_steps=4 (effective batch 32) and max_seq_length=512, which requires gradient_checkpointing=true — without it, this config OOMs at ~12.5GB on a T4 because per-step tokens jump from ~200 (smoke test) to 4,096 (real config), and activation memory without checkpointing scales heavily with both batch size and sequence length. Do not increase batch size or sequence length without also confirming gradient_checkpointing is still enabled.
