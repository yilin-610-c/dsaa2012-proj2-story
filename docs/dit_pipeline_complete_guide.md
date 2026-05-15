# DiT Pipeline 完全指南

> 从 SDXL U-Net 到 PixArt-α DiT Transformer 的 Story Image Generation Pipeline
>
> 最后更新: 2026-05-14

---

## 目录

1. [快速开始：跑一个故事的完整命令](#1-快速开始跑一个故事的完整命令)
2. [Pipeline 架构概览](#2-pipeline-架构概览)
3. [所有可用 Profile](#3-所有可用-profile)
4. [LoRA DreamBooth 训练流程](#4-lora-dreambooth-训练流程)
5. [Story-Joint 跨场景融合机制](#5-story-joint-跨场景融合机制)
6. [调参指南](#6-调参指南)
7. [已知限制与改进方向](#7-已知限制与改进方向)
8. [文件清单](#8-文件清单)

---

## 1. 快速开始：跑一个故事的完整命令

### 1.0 一键全流程 ⭐ (推荐)

```bash
bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt
```

一条命令 = 生成参考图 + 训练 LoRA + 推理。输出在 `outputs/dit_lora_smoke_full_<time>_<story>/`。Step 1 默认用 **PixArt-α** 生成训练图（与 Step 2/3 同一 backbone）；旧行为可加 `--ref-backend sdxl`。

推理加载的 LoRA：脚本优先使用 `lora_checkpoints/` 目录根下的最终 `adapter_model.safetensors`（训练脚本结束时写入），仅在没有根 adapter 时才回退到最新的 `checkpoint-*`。这样不会在用「非整除 checkpoint 步数」的训练时误加载少训一段的权重，避免生成人物与训练图脱节。

### 1.1 Baseline (SDXL, 原始)

```bash
PYTHONPATH=src python -m storygen.cli --profile smoke_test --input test_set/01.txt
```

### 1.2 PixArt Scene-Level (独立生成)

```bash
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test --input test_set/01.txt
```

### 1.3 PixArt Scene-Level + LoRA (推荐) ⭐

```bash
PYTHONPATH=src python -m storygen.cli \
  --profile dit_lora_smoke \
  --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-800 \
  --set model.lora_trigger="sks "
```

### 1.4 PixArt Story-Joint (跨场景融合)

```bash
PYTHONPATH=src python -m storygen.cli \
  --profile dit_story_joint_smoke \
  --input test_set/01.txt
```

### 1.5 PixArt Story-Joint + LoRA

```bash
PYTHONPATH=src python -m storygen.cli \
  --profile dit_story_joint_lora \
  --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-800 \
  --set model.lora_trigger="sks " \
  --set model.story_attention.blend_strength=0.05 \
  --set model.story_attention.cross_scene_layers=[12] \
  --set model.width=512 --set model.height=512 --set model.num_inference_steps=8
```

### 1.6 A/B 对比测试

```bash
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test        --input test_set/01.txt
PYTHONPATH=src python -m storygen.cli --profile dit_lora_smoke        --input test_set/01.txt --set model.lora_path=... --set model.lora_trigger="sks "
```

---

## 2. Pipeline 架构概览

```
故事文本(.txt) → Parser → Prompt Builder → Generator Factory → Generator → 图像输出
  │
  ├── diffusers_text2img (SDXL UNet, scene-level)
  ├── dit_text2img       (PixArt DiT, scene-level)
  ├── storydiffusion_direct (SDXL proxy, story-level)
  └── dit_story_joint    (PixArt DiT, story-level ★)
```

### Backend 对比

| Backend | 粒度 | 模型 | 跨场景 | 角色一致性 |
|---------|------|------|:---:|:---:|
| `diffusers_text2img` | scene | SDXL UNet | ❌ | ✅ IP-Adapter |
| `dit_text2img` | scene | PixArt-α DiT | ❌ | ✅ LoRA |
| `storydiffusion_direct` | story | SDXL (代理) | ❌ | ✅ IP-Adapter |
| `dit_story_joint` | story | PixArt-α DiT | ✅ | ✅ LoRA |

### Story-Joint 内部流程

```
PixArtAlphaPipeline.__call__(prompt=[p1, p2, p3], ...)
  │
  ├── T5 Encoder: 3 prompts → (3, seq, dim)
  ├── Latents: randn(3, C, H, W)   ← 所有场景在一个 batch
  │
  ├── Denoising loop:
  │     CFG: latents → (6, C, H, W)
  │     DiT Transformer (28 blocks):
  │       Block 0-11:  标准 self-attention
  │       Block 12:  ★ CrossSceneAttention (global context blend)
  │       Block 13-23: 标准 self-attention
  │       Block 24:  ★ CrossSceneAttention (global context blend)
  │       Block 25-27: 标准 self-attention
  │
  ├── VAE Decode: 每场景独立 → 3 张 PIL images
  └── ImageProcessor postprocess
```

### CrossSceneSelfAttentionWrapper 机制

```python
# 全局上下文对齐（非空间位置对齐）
x = hidden_states.reshape(groups, N, T, D)

# 每场景池化为一个"风格向量"
scene_global = x.mean(dim=T)          # (groups, N, 1, D)

# 跨场景平均 → 统一色调偏置
cross_global = scene_global.mean(dim=N)  # (groups, 1, 1, D)

# 所有 token 加相同的偏置（保护空间结构，尤其移动角色）
x_blended = (1-α)*x + α*cross_global

# 标准 self-attention
return self.attn(x_blended, ...)
```

关键：不再按空间位置对齐 → 角色移动时脸部 token 不会被错误混合。

---

## 3. 所有可用 Profile

### Scene-Level

| Profile | 分辨率 | Steps | CFG | LoRA | 用途 |
|---------|--------|-------|-----|:---:|------|
| `smoke_test` | 384² | 2 | 0.0 | ❌ | SDXL 快速测试 |
| `dit_smoke_test` | 512² | 8 | 4.5 | ❌ | PixArt 快速测试 |
| `dit_full_quality` | 1024² | 20 | 4.5 | ❌ | PixArt 高质量 |
| `dit_lora_smoke` | 512² | 8 | 4.5 | ✅ | PixArt + LoRA (推荐) |

### Story-Level (跨场景融合)

| Profile | Layers | Blend | Per-layer | LoRA | 用途 |
|---------|--------|-------|:---:|:---:|------|
| `dit_story_joint_light` | [12,24] | 0.05 | ❌ | ❌ | 最轻融合 |
| `dit_story_joint_smoke` | [12,24] | 0.15 | ❌ | ❌ | 默认 (512², 8步) |
| `dit_story_joint_medium` | [6,12,18,24] | 0.15 | ❌ | ❌ | 4层均匀 |
| `dit_story_joint_strong` | [4,8,12,16,20,24] | (per-layer) | ✅ | ❌ | 分层控制 |
| `dit_story_joint_lora` | [12,24] | 0.15 | ❌ | ✅ | 全功能 (768², 20步) |

### `dit_story_joint_strong` 的分层配置

```yaml
blend_strength_per_layer:
  4: 0.35    # 极浅层: 纹理融合（强）
  8: 0.30    # 浅层:   颜色融合（较强）
  12: 0.20   # 中层:   形状融合（中等）
  16: 0.15   # 中深层: 弱融合
  20: 0.05   # 深层:   极弱
  24: 0.0    # 极深层: 关闭（保护角色身份）
```

---

## 4. LoRA DreamBooth 训练流程

### 4.1 一键运行 ⭐

```bash
bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt
```

一条命令跑完 Step 1→2→3。所有输出统一到:

```
outputs/dit_lora_smoke_full_<timestamp>_<story>/
├── training_images/          ← Step 1: 默认 PixArt-α 参考图 (~16)；可选 --ref-backend sdxl 走旧版 SDXL+IP-Adapter
├── lora_checkpoints/         ← Step 2: 根目录 final adapter + checkpoint-*/
├── generation/               ← Step 3: 故事生成结果
│   └── scenes/
│       ├── scene_001/selected.png
│       ├── scene_002/selected.png
│       └── scene_003/selected.png
└── run.log                   ← 完整日志
```

可选参数:

```bash
bash scripts/run_full_lora_pipeline.sh \
  --input test_set/01.txt \
  --profile dit_lora_smoke \
  --ref-backend pixart \
  --lora-rank 16 \
  --lora-steps 800
```

### 4.2 分步运行（调试用）

```bash
# Step 1: 生成训练参考图（默认 PixArt-α，与 LoRA 训练/推理同一 backbone；legacy 加 --ref-backend sdxl）
PYTHONPATH=src python scripts/gen_lora_ref_images.py \
  --input test_set/01.txt \
  --output-dir training_data/my_character \
  --ref-backend pixart

# Step 2: PixArt LoRA DreamBooth 训练 (~4分钟)
/home/xyz/.conda/envs/ipadapter/bin/python -m accelerate.commands.launch \
  /home/xyz/Desktop/xluo/StoryDiffusion/dsaa2012-proj2-story/scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
  --train_data_dir=/home/xyz/Desktop/xluo/StoryDiffusion/dsaa2012-proj2-story/training_data/my_character \
  --output_dir=/home/xyz/Desktop/xluo/StoryDiffusion/dsaa2012-proj2-story/lora_checkpoints/character \
  --resolution=512 --rank=16 --train_batch_size=1 \
  --learning_rate=1e-06 --max_train_steps=800 \
  --checkpointing_steps=200 --gradient_checkpointing --mixed_precision=fp16

# Step 3: 推理
PYTHONPATH=src python -m storygen.cli \
  --profile dit_lora_smoke \
  --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-800 \
  --set model.lora_trigger="sks "
```

### 4.2 训练输出

```
lora_checkpoints/character/
├── adapter_model.safetensors  ← 训练结束时的最终权重（推理优先用这个路径）
├── adapter_config.json
├── checkpoint-200/           ← 中间 checkpoint (也可推理)
├── checkpoint-400/
├── checkpoint-600/
└── checkpoint-800/           ← 最后一次中间落盘；若 max_train_steps 不是 checkpointing_steps 的整数倍，最后一段只存在于根目录 adapter
```

### 4.3 训练数据

```
training_data/my_character/
├── {Name}_canonical.png       ← Phase 1: anchor_bank 选出的 canonical（默认 PixArt；sdxl 后端则为 SDXL）
├── {Name}_diverse_000.png …   ← Phase 2: 多样化 prompt（默认 PixArt text2img；sdxl 为 IP-Adapter + SDXL）
└── metadata.jsonl             ← HuggingFace ImageFolder 格式
```

### 4.4 训练参数

| 参数 | v1 (旧) | v2 (当前) | 说明 |
|------|---------|-----------|------|
| 训练图像 | 4 张 (仅锚定) | 13 张 (多样) | 更多样化 |
| rank | 8 | 16 | 更多参数 |
| steps | 500 | 800 | 更多迭代 |
| adapter 大小 | 27MB | 55MB | - |

### 4.5 推理必须带 `lora_trigger`

```bash
# ✅ 正确: 有 trigger
--set model.lora_trigger="sks "
# prompt 变成: "sks Lily, makes breakfast..."

# ❌ 错误: 无 trigger → LoRA 不激活
# prompt 是: "Lily, makes breakfast..."  ← LoRA 不会生效！
```

**风格对齐（重要）**：默认训练参考图由 **PixArt** 生成，与 LoRA 训练/推理同一 backbone。若使用 `--ref-backend sdxl`，训练图来自 SDXL，caption 仍偏 photorealistic；推理 profile 已用 photorealistic 向 `style_prompt` 对齐。

---

## 5. Story-Joint 跨场景融合机制

### 5.1 设计原则

- **全局上下文对齐**（非空间位置对齐）：只加全局偏置，不改变空间结构
- **分层控制**：浅层强融合（纹理/颜色），深层弱融合（语义/身份）
- **零额外参数**：纯 blending，兼容所有 attention backend

### 5.2 为什么 Scene-Level 可能比 Story-Joint 好

当使用 LoRA 时，以下因素可能导致 story-joint 效果不如 scene-level：

| 因素 | 影响 |
|------|------|
| **深层 blending 冲淡 LoRA** | LoRA 特征在深层（layer 24）生效，但 blending 也在深层做融合 → LoRA 的身份特征被 15% 的跨场景偏置覆盖 |
| **分辨率差异** | story-joint 用 768² (2304 tokens)，scene-level 用 512² (1024 tokens) → 更多 token 被 blending 影响 |
| **更多步数** | 20 steps vs 8 steps → LoRA 的小偏差在每步累积 |
| **Batch 噪声共享** | 所有场景共享初始 latent 结构 → 跨场景 blending 让噪声渗入其他场景 |
| **角色-故事 mismatch** | LoRA 在 01.txt 的 Lily 上训练，但用到 02.txt 的 Ryan → trigger token 有冲突 |

### 5.3 优化 Story-Joint + LoRA 的建议

```bash
# 降低 blending 强度
--set model.story_attention.blend_strength=0.05

# 只在中层融合，不在深层
--set model.story_attention.cross_scene_layers=[6,12]

# 降低分辨率匹配 scene-level
--set model.width=512 --set model.height=512

# 减少步数
--set model.num_inference_steps=8
```

---

## 6. 调参指南

### 核心参数

| 参数 | 默认 | 范围 | 效果 |
|------|------|------|------|
| `blend_strength` | 0.15 | 0.0-0.5 | 跨场景融合强度 |
| `cross_scene_layers` | [12,24] | [0..27] | 启用融合的层 |
| `blend_strength_per_layer` | 无 | dict | 每层独立强度 |
| `guidance_scale` | 4.5 | 1.0-7.0 | CFG |
| `num_inference_steps` | 8 | 4-50 | 去噪步数 |
| `rank` (LoRA) | 16 | 4-64 | LoRA 秩 |
| `lora_trigger` | 无 | string | DreamBooth 触发词 |

### blend_strength 经验值

| 值 | 效果 |
|----|------|
| 0.0 | 无融合 = 独立生成 |
| 0.05-0.10 | 极轻微，安全区 |
| 0.15-0.25 | 推荐区间 |
| 0.30-0.50 | 激进，可能过平滑 |

### 浅层 vs 深层

| 层级 | 影响特征 | LoRA 兼容性 |
|------|---------|:---:|
| 0-8 (浅) | 纹理、颜色、光照 | ✅ 兼容 |
| 9-18 (中) | 形状、构图 | ⚠️ 谨慎 |
| 19-27 (深) | 语义、身份、人脸 | ❌ 建议关闭 |

---

## 7. 已知限制与改进方向

### 当前限制

| # | 限制 | 影响 |
|---|------|------|
| 1 | Story-Joint + LoRA 可能冲突 | 深层 blending 冲淡 LoRA 身份特征 |
| 2 | LoRA 仅绑一个角色 | 每角色需独立训练 |
| 3 | PixArt 无 IP-Adapter | 不能直接输入参考图 |
| 4 | 训练分辨率 512² | 低于原生 1024² |
| 5 | 全局上下文不传空间信息 | 背景一致性提升有限 |

### 改进方向

| 优先级 | 方向 | 预期效果 |
|:---:|------|------|
| **高** | 训练时增加更多全身/动作图 | 角色在动态场景中更一致 |
| **高** | 调低 story-joint 深层 blend=0 | LoRA + 跨场景互不冲突 |
| **中** | 语义对齐 blending (similarity-based) | 背景一致性 + 角色保护 |
| **中** | 1024² 训练分辨率 | 更高质量 |
| **低** | DiT 版 IP-Adapter (研究级) | 端到端图片条件控制 |

---

## 8. 文件清单

### 新增 (本项目贡献)

| 文件 | 说明 |
|------|------|
| `src/storygen/generators/dit_text2img.py` | Scene-level PixArt-α generator + LoRA |
| `src/storygen/generators/dit_story_joint.py` | Story-level PixArt-α joint + CrossSceneAttention + LoRA |
| `scripts/train_pixart_lora_hf.py` | PixArt DreamBooth LoRA 训练 (官方) |
| `scripts/gen_lora_ref_images.py` | 训练参考图：默认 PixArt-α；`--ref-backend sdxl` 为旧版 SDXL+IP-Adapter |
| `docs/dit_pipeline_complete_guide.md` | 本文档 |
| `docs/dit_full_implementation_report.md` | 完整实现报告 |

### 修改 (本项目贡献)

| 文件 | 改动 |
|------|------|
| `src/storygen/generators/factory.py` | 注册 `dit_text2img` + `dit_story_joint` |
| `configs/base.yaml` | 新增 ~12 个 DiT/LoRA profile |
| `.gitmodules` | facebookresearch/DiT 子模块 |

### 新增依赖

```bash
pip install peft tiktoken sentencepiece protobuf datasets torchvision
```

示例：

```bash
bash scripts/run_full_lora_pipeline.sh --input test_set/02.txt
```