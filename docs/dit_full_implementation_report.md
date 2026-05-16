# DiT + DreamBooth LoRA 完整实现报告

> StoryDiffusion Pipeline: 从 SDXL U-Net 到 PixArt-α DiT Transformer 的迁移
>
> 日期: 2026-05-13 | 分支: `integrate/prompt-audit-into-storydiffusion`

---

## 目录

1. [项目背景](#1-项目背景)
2. [原始架构](#2-原始架构)
3. [DiT 改造总览](#3-dit-改造总览)
4. [Level 1: Scene-level DiT Generator](#4-level-1-scene-level-dit-generator)
5. [Level 3: Story-level Joint DiT Generator](#5-level-3-story-level-joint-dit-generator)
6. [CrossSceneSelfAttentionWrapper 详解](#6-crosssceneselfattentionwrapper-详解)
7. [DreamBooth LoRA 角色一致性方案](#7-dreambooth-lora-角色一致性方案)
8. [完整文件变更清单](#8-完整文件变更清单)
9. [Config Profile 速查表](#9-config-profile-速查表)
10. [命令速查](#10-命令速查)
11. [调参指南](#11-调参指南)
12. [已知问题与解决](#12-已知问题与解决)
13. [技术决策记录](#13-技术决策记录)

---

## 1. 项目背景

### 1.1 任务目标

根据 `project_requirements.md`:

> 输入多场景（multi-panel）文本描述，输出一系列满足以下条件的图像:
> - **Per-panel correctness**: 每张图精确匹配场景描述
> - **Cross-panel consistency**: 角色、背景、风格跨场景一致
> - **Narrative continuity**: 场景间自然过渡

### 1.2 课程约束

- 固定的、可复现的自动化 pipeline
- 禁止手工逐 case 编辑
- 禁止 hard-code 特定 test case 输出
- 禁止 agent 系统和外部 API
- 教师 + TA 人工评估，对比其他组的**相对质量**

---

## 2. 原始架构

### 2.1 核心链路

```
故事文本(.txt) → Parser → Prompt Builder → Generator → Scoring → 图像输出
```

### 2.2 原始 Backend

| Backend | 模型 | 粒度 | 角色一致性 | 场景一致性 |
|---------|------|------|-----------|-----------|
| `diffusers_text2img` | SDXL/SDXL-Turbo (UNet) | scene | ✅ IP-Adapter | ❌ 独立生成 |
| `storydiffusion_direct` | SDXL-Turbo (代理) | story | ✅ IP-Adapter | ❌ 仅代理模式 |

### 2.3 原始问题

- SDXL-Turbo 基于 U-Net 架构，不能利用 Transformer 的 attention 灵活性
- 跨场景一致性仅靠 prompt 质量和相似 seed
- IP-Adapter 虽好，但仅针对 U-Net cross-attention 设计

---

## 3. DiT 改造总览

### 3.1 为什么选 PixArt-α

| 候选模型 | 架构 | Text-to-Image | Diffusers 支持 | 训练生态 |
|---------|------|:---:|:---:|:---:|
| facebook/DiT | ViT | ❌ (仅 class-cond) | ❌ | ❌ |
| **PixArt-α** | DiT + T5 cross-attn | ✅ | ✅ PixArtAlphaPipeline | ✅ LoRA/DreamBooth |
| SD3 | MM-DiT | ✅ | ✅ | ✅ LoRA |
| FLUX | Hybrid | ✅ | ✅ | ✅ LoRA |

选 PixArt-α 是因为: 纯 DiT 架构、diffusers 原生支持、官方 DreamBooth LoRA 训练脚本。

### 3.2 三层改造

```
Level 1 (基础设施):  dit_text2img       → scene-level PixArt-α 生成
Level 3 (核心创新):  dit_story_joint    → story-level 跨场景融合 + CrossSceneAttention
LoRA  (角色锚定):    DreamBooth LoRA    → IP-Adapter 参考图 → LoRA 训练 → 推理叠加
```

### 3.3 最终 Backend 矩阵

| Backend | 粒度 | 模型 | 跨场景 | 角色 | 文件 |
|---------|------|------|:---:|:---:|------|
| `diffusers_text2img` | scene | SDXL UNet | ❌ | ✅ IP-Adapter | `diffusers_text2img.py` |
| `storydiffusion_direct` | story | SDXL (代理) | ❌ | ✅ IP-Adapter | `storydiffusion_direct.py` |
| **`dit_text2img`** | scene | PixArt-α DiT | ❌ | ✅ LoRA | `dit_text2img.py` ⭐ |
| **`dit_story_joint`** | story | PixArt-α DiT | ✅ | ✅ LoRA | `dit_story_joint.py` ⭐ |

---

## 4. Level 1: Scene-level DiT Generator

### 4.1 文件

`src/storygen/generators/dit_text2img.py` (~110行)

### 4.2 架构

```python
class DitTextToImageGenerator(BaseImageGenerator):
    def load(self):
        # 1. 加载 PixArtAlphaPipeline
        pipeline = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
        )
        # 2. 可选: 加载 LoRA (必须在 CPU offload 之前)
        if lora_path:
            pipeline.transformer = PeftModel.from_pretrained(...)
            pipeline.transformer = pipeline.transformer.merge_and_unload()
        # 3. CPU offload 优化显存
        pipeline.enable_model_cpu_offload()

    def generate_scene(self, request):
        # 每场景独立 pipeline.__call__(prompt=...)
        result = pipeline(prompt=request.prompt_spec.generation_prompt, ...)
        return GenerationCandidate(image=result.images[0], ...)
```

### 4.3 与 SDXL Generator 的差异

| | DiffusersTextToImageGenerator | DitTextToImageGenerator |
|---|---|---|
| Pipeline | `AutoPipelineForText2Image` | `PixArtAlphaPipeline` |
| 文本编码器 | CLIP (SDXL) | T5 (PixArt) |
| 去噪模型 | UNet | DiT Transformer (28 blocks) |
| 注意机制 | UNet cross-attn | DiT adaLN-single + cross-attn |
| IP-Adapter | ✅ | ❌ (不支持) |
| LoRA | ❌ | ✅ |
| 推荐 CFG | 0.0 (Turbo) | 4.5 |
| 推荐 Steps | 4 | 8-20 |

---

## 5. Level 3: Story-level Joint DiT Generator

### 5.1 文件

`src/storygen/generators/dit_story_joint.py` (~350行)

### 5.2 架构

```python
class DitStoryJointGenerator(BaseStoryGenerator):
    def load(self):
        # 1. 加载 PixArtAlphaPipeline
        # 2. 可选: 加载 LoRA (merge_and_unload)
        # 3. CPU offload

    def generate_story(self, request):
        num_scenes = len(request.scene_plans)

        # Step 1: 注入 CrossSceneSelfAttentionWrapper
        self._inject_cross_scene_attention(num_scenes)
        self._update_num_scenes(num_scenes)

        # Step 2: 所有场景 prompt 作为一个 list 传给 pipeline
        prompts = [plan.generation_prompt for plan in request.scene_plans]
        result = pipeline(prompt=prompts, ...)
        # ↑ pipeline 内部: 所有 N 个场景在一个 batch 中去噪
        # ↑ CrossSceneSelfAttentionWrapper 在指定层自动做跨场景融合

        # Step 3: 拆分为 N 个 PanelGenerationOutput
        for i, plan in enumerate(request.scene_plans):
            panel_outputs.append(PanelGenerationOutput(image=result.images[i], ...))
```

### 5.3 关键：Pipeline 的 Batch Inference

```
PixArtAlphaPipeline.__call__(prompt=[p1, p2, p3], ...)

  内部流程:
  ┌──────────────────────────────────────────────────────────┐
  │ T5 Encoder: encode 3 prompts → (3, seq_len, dim)        │
  │ Latents: randn(3, C, H, W)                               │
  │                                                          │
  │ Denoising loop:                                          │
  │   CFG: latents → (6, C, H, W)                            │
  │   DiT Transformer (28 blocks):                           │
  │     Block 0-3:  标准 self-attention                     │
  │     Block 4:  ★ CrossSceneAttention (blend=0.35)        │
  │     ...                                                   │
  │     Block 24: ★ CrossSceneAttention (blend=0.0)         │
  │     Block 25-27: 标准 self-attention                    │
  │   Scheduler step                                         │
  │                                                          │
  │ VAE Decode: 每场景独立 decode → 3 张 PIL images         │
  └──────────────────────────────────────────────────────────┘
```

### 5.4 与 Scene-level 的关键区别

| | Scene-level (dit_text2img) | Story-level (dit_story_joint) |
|---|---|---|
| Pipeline 调用次数 | N 次 | **1 次** |
| Latent 形状 | (1, C, H, W) × N | **(N, C, H, W)** |
| CFG 后 batch | (2, C, H, W) × N | **(2N, C, H, W)** |
| 跨场景信息流 | 无 | **CrossSceneAttention** |
| 速度 | ~6秒/场景 × N | ~10秒 total (N=3, 8 steps) |

---

## 6. CrossSceneSelfAttentionWrapper 详解

### 6.1 设计动机

原始方案（空间位置对齐 — 已废弃）:
```
x_mean = x.mean(dim=scenes)           # 每空间位置跨场景平均
x_blended[i] = (1-α)*x[i] + α*x_mean
```
问题: 角色移动到不同位置 → 脸部 token 混入其他场景的背景 token → 角色身份破坏。

### 6.2 当前方案（全局上下文对齐）

```python
def forward(self, hidden_states, ...):
    # hidden_states: (B, T, D)  B=groups×N  groups={1,2(CFG)}
    x = hidden_states.reshape(groups, N, T, D)   # 拆分场景

    # 每场景池化为一个"风格向量"
    scene_global = x.mean(dim=T, keepdim=True)    # (groups, N, 1, D)

    # 跨场景风格平均 → 统一调色板
    cross_global = scene_global.mean(dim=N, keepdim=True)  # (groups, 1, 1, D)

    # 每个 token 加相同的全局偏置
    x_blended = (1-α)*x + α*cross_global

    # 标准 self-attention
    return self.attn(x_blended.reshape(B, T, D), ...)
```

### 6.3 为什么保护角色

```
旧方案（空间对齐）:
  场景A 位置(16,24): [👤 角色脸部]  ←→  场景B 位置(16,24): [🪵 桌子]
  → 平均值 = 无意义的混合 → 角色被污染

新方案（全局上下文）:
  场景A 所有 token → 池化 → [色调偏置向量]
  场景B 所有 token → 池化 → [色调偏置向量]
  → 平均值 = 统一的色调偏移
  → 每个 token 只加全局偏置 → 空间结构（人脸形状）完整保留
```

### 6.4 分层 Blend Strength

```yaml
# dit_story_joint_strong 的 per-layer 配置
story_attention:
  cross_scene_layers: [4, 8, 12, 16, 20, 24]
  blend_strength_per_layer:
    4: 0.35    # 极浅层: 纹理融合（强）
    8: 0.30    # 浅层:   颜色融合（较强）
    12: 0.20   # 中层:   形状融合（中等）
    16: 0.15   # 中深层: 弱融合
    20: 0.05   # 深层:   极弱融合
    24: 0.0    # 极深层: 关闭（保护角色身份语义）
```

Transformer 层级与特征对应:
- Layers 0-3:  纹理 (边缘、噪点)
- Layers 4-11: 颜色 (色调、饱和度)
- Layers 12-19: 形状 (物体轮廓、构图)
- Layers 20-27: 语义 (物体身份、人脸)

---

## 7. DreamBooth LoRA 角色一致性方案

### 7.1 为什么不用 IP-Adapter

| | SDXL Pipeline | PixArtAlphaPipeline |
|---|---|---|
| `load_ip_adapter()` | ✅ `IPAdapterMixin` | ❌ 不存在 |
| IP-Adapter 权重 | `h94/IP-Adapter` (SDXL) | 不存在 (社区未训练) |
| 架构兼容 | UNet cross-attention | DiT adaLN-single |

### 7.2 LoRA 方案流程

```
┌──────────────────────────────────────────────────────────┐
│                LoRA 训练闭环                             │
│                                                          │
│  Step 1: 生成参考图                                      │
│    scripts/gen_lora_ref_images.py                        │
│    使用 SDXL anchor_bank + IP-Adapter                    │
│    → 同角色多角度参考图 (5-10张)                         │
│                                                          │
│  Step 2: PixArt LoRA DreamBooth 训练                     │
│    scripts/train_pixart_lora_hf.py (官方脚本)            │
│    → 27MB LoRA adapter (rank=8)                          │
│    → 500 steps × 4 steps/sec ≈ 2分钟                    │
│                                                          │
│  Step 3: LoRA + Story-Joint 推理                         │
│    dit_story_joint + lora_path=checkpoint-500            │
│    → LoRA 锚定角色 + CrossSceneAttention 统一场景       │
└──────────────────────────────────────────────────────────┘
```

### 7.3 LoRA 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| resolution | 512 | 降低显存 (原1024 OOM) |
| rank | 8 | LoRA 秩 (原16) |
| train_batch_size | 1 | 降低显存 (原4 OOM) |
| learning_rate | 1e-6 | 标准 DreamBooth LR |
| max_train_steps | 500 | 限制训练步数 |
| mixed_precision | fp16 | 半精度 |
| trainable params | 6,905,216 (1.1%) | 仅 1% 参数可训练 |
| adapter size | 27MB | 轻量级 |

### 7.4 LoRA 加载实现

```python
# dit_text2img.py / dit_story_joint.py

def _load_lora(self, pipeline, lora_path):
    from peft import PeftModel
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer, lora_path
    )
    # merge_and_unload() 将 LoRA 权重永久 bake 进 transformer
    # 必须在 enable_model_cpu_offload() 之前调用
    pipeline.transformer = pipeline.transformer.merge_and_unload()
```

### 7.5 训练输出

```
lora_checkpoints/character/
├── checkpoint-100/           ← 中间 checkpoint (都可推理)
├── checkpoint-200/
├── checkpoint-300/
├── checkpoint-400/
├── checkpoint-500/           ← 推荐用最新的
├── adapter_model.safetensors  (27MB)
├── adapter_config.json
└── train.log
```

---

## 8. 完整文件变更清单

### 8.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/storygen/generators/dit_text2img.py` | ~120 | Scene-level PixArt-α generator + LoRA |
| `src/storygen/generators/dit_story_joint.py` | ~360 | Story-level PixArt-α joint generator + CrossSceneAttention + LoRA |
| `scripts/train_pixart_lora_hf.py` | 1042 | PixArt DreamBooth LoRA 训练脚本 (官方) |
| `scripts/gen_lora_ref_images.py` | ~190 | SDXL anchor_bank + IP-Adapter → LoRA 训练参考图 |
| `docs/dit_pipeline_complete_guide.md` | ~400 | Pipeline 架构完整指南 |
| `docs/dit_full_implementation_report.md` | 本文档 | 完整实现报告 |
| `third_party/facebookresearch-DiT/` | submodule | 原始 DiT 代码参考 |

### 8.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/storygen/generators/factory.py` | 注册 `dit_text2img` + `dit_story_joint` backend |
| `configs/base.yaml` | 新增 `max_sequence_length`, `story_attention`, `lora_path`, `blend_strength_per_layer`; 新增 10 个 profile |
| `.gitmodules` | 添加 `facebookresearch/DiT` 子模块 |

### 8.3 新增依赖

```bash
pip install peft tiktoken sentencepiece protobuf datasets torchvision
```

### 8.4 10 个新增 Profile

| # | Profile | Backend | 特点 |
|---|---------|---------|------|
| 1 | `dit_smoke_test` | dit_text2img | PixArt scene-level 快速测试 |
| 2 | `dit_full_quality` | dit_text2img | PixArt 高质量 (1024², 20步) |
| 3 | `dit_story_joint_smoke` | dit_story_joint | 默认融合 (blend=0.15, layers=[12,24]) |
| 4 | `dit_story_joint_full` | dit_story_joint | 高质量融合 (768², 20步) |
| 5 | `dit_story_joint_light` | dit_story_joint | 最轻融合 (blend=0.05) |
| 6 | `dit_story_joint_medium` | dit_story_joint | 4层均匀融合 |
| 7 | `dit_story_joint_strong` | dit_story_joint | 6层 per-layer 分层控制 |
| 8 | `dit_lora_smoke` | dit_text2img | PixArt + LoRA |
| 9 | `dit_story_joint_lora` | dit_story_joint | 全功能: LoRA + Story-Joint |
| 10 | `dit_full_quality` | dit_text2img | 高质量 scene-level |

---

## 9. Config Profile 速查表

### Scene-level

```bash
# SDXL baseline (原始)
PYTHONPATH=src python -m storygen.cli --profile smoke_test --input test_set/01.txt

# PixArt baseline
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test --input test_set/01.txt

# PixArt + LoRA
PYTHONPATH=src python -m storygen.cli --profile dit_lora_smoke --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-500
```

### Story-level (跨场景融合)

```bash
# 轻量
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_light --input test_set/01.txt

# 默认
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_smoke --input test_set/01.txt

# 强融合（分层控制）
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_strong --input test_set/01.txt

# 全功能（LoRA + Story-Joint）
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_lora --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-500
```

### Per-layer blend_strength 配置示例

```yaml
# dit_story_joint_strong
story_attention:
  enabled: true
  cross_scene_layers: [4, 8, 12, 16, 20, 24]
  blend_strength_per_layer:
    4: 0.35    # 纹理 → 强融合
    8: 0.30    # 颜色 → 较强
    12: 0.20   # 形状 → 中等
    16: 0.15   # 中深层 → 弱
    20: 0.05   # 深层语义 → 极弱
    24: 0.0    # 身份 → 关闭
```

---

## 10. 命令速查

### 基础生成

```bash
# Scene-level PixArt
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test --input test_set/01.txt

# Story-level PixArt (跨场景融合)
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_smoke --input test_set/01.txt

# 临时调参
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_smoke \
  --input test_set/01.txt \
  --set model.story_attention.blend_strength=0.25 \
  --set model.story_attention.cross_scene_layers=[4,8,12,16,20,24]
```

### LoRA 训练

```bash
# Step 1: 生成参考图
PYTHONPATH=src python scripts/gen_lora_ref_images.py \
  --input test_set/01.txt \
  --output-dir training_data/my_character

# Step 2: 训练 LoRA
/home/xyz/.conda/envs/ipadapter/bin/python -m accelerate.commands.launch \
  /absolute/path/to/scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
  --train_data_dir=/absolute/path/to/training_data/my_character \
  --output_dir=/absolute/path/to/lora_checkpoints/character \
  --resolution=512 --rank=8 --train_batch_size=1 \
  --learning_rate=1e-06 --num_train_epochs=200 \
  --max_train_steps=500 --checkpointing_steps=100 \
  --gradient_checkpointing --mixed_precision=fp16

# Step 3: 推理
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_lora \
  --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-500
```

### A/B 对比测试

```bash
# 4 个 profile 对比 → outputs/ 下的 4 个 run 目录
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test            --input test_set/01.txt
PYTHONPATH=src python -m storygen.cli --profile dit_lora_smoke            --input test_set/01.txt --set model.lora_path=...
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_smoke     --input test_set/01.txt
PYTHONPATH=src python -m storygen.cli --profile dit_story_joint_lora      --input test_set/01.txt --set model.lora_path=...
```

---

## 11. 调参指南

### 11.1 核心参数

| 参数 | 默认 | 范围 | 效果 |
|------|------|------|------|
| `blend_strength` | 0.15 | 0.0-0.5 | 跨场景融合强度 |
| `cross_scene_layers` | [12,24] | [0..27] | 启用融合的层 |
| `blend_strength_per_layer` | 无 | dict | 每层独立强度 |
| `guidance_scale` | 4.5 | 1.0-7.0 | CFG 引导强度 |
| `num_inference_steps` | 8 | 4-50 | 去噪步数 |
| `rank` (LoRA) | 8 | 4-64 | LoRA 秩 |
| `lora_path` | null | path | LoRA checkpoint |

### 11.2 blend_strength 经验值

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.0 | 无融合 = 独立生成 | baseline |
| 0.05-0.10 | 极轻微色调统一 | 保守实验 |
| 0.15-0.25 | 推荐区间 | 日常使用 |
| 0.30-0.50 | 强融合 | 需要高度统一风格 |

### 11.3 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 图片是乱码 | 旧自定义 denoising loop + resolution 不匹配 | 已修复: 改用 pipeline.__call__ |
| 跨场景没效果 | blend=0 或 layers 不对 | 检查 config |
| 角色被洗掉 | 空间位置对齐 | 已修复: 全局上下文对齐 |
| OOM @ 训练 | 1024² + batch=4 | 降为 512² + batch=1 + rank=8 |
| LoRA 加载报 _hf_hook | LoRA 在 offload 之后加载 | 已修复: LoRA 优先加载 |
| tokenizer 报错 | 缺少依赖 | `pip install tiktoken sentencepiece protobuf` |
| 加速器用系统 Python | `~/.local/bin` 优先 | 用 conda 完整路径 |

---

## 12. 已知问题与解决

### 12.1 已解决

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | 生成图片是乱码 | 自定义 denoising loop 用 64×64 latent 喂给期望 128×128 的 transformer | 改用 `pipeline.__call__()` |
| 2 | `added_cond_kwargs` None error | 手写 loop 漏传 resolution/aspect_ratio | 改用 `pipeline.__call__()` |
| 3 | dtype mismatch (Half vs Float) | mixer 创建时未匹配模型 dtype | 修复 `.to(dtype=param.dtype)` |
| 4 | 角色一致性差 | 空间位置对齐 blending 错误匹配 token | 改为全局上下文对齐 |
| 5 | OOM @ 训练 | PixArt-XL + 1024² + batch=4 | 降为 512² + batch=1 + rank=8 |
| 6 | LoRA + CPU offload 冲突 | merge_and_unload 后 hook 丢失 | LoRA 在 offload 前加载 |
| 7 | accelerate 用系统 Python | PATH 优先级 | 使用 conda 完整路径 |
| 8 | metadata.jsonl 格式错误 | `image` key 但 ImageFolder 需 `file_name` | 修复 key |
| 9 | 训练 dataset_name 错误 | 本地数据需 `--train_data_dir` | 修复 flag |

### 12.2 当前限制

| # | 限制 | 影响 | 可能的改进 |
|---|------|------|-----------|
| 1 | LoRA 仅 4 张参考图 | 角色泛化可能不足 | 增加参考图数量/多样性 |
| 2 | 全局上下文不传空间信息 | 背景一致性提升有限 | 添加 semantic-similarity-based blending |
| 3 | PixArt 无 IP-Adapter | 不能直接用参考图推理 | 需社区训练 PixArt IP-Adapter |
| 4 | 训练分辨率 512² | 低于模型原生 1024² | 使用更大显存 GPU |
| 5 | 跨场景仅做 self-attn 前 blending | 不修改 attention 计算本身 | 改为 cross-scene KV sharing |

---

## 13. 技术决策记录

### 决策 1: 为什么选 PixArt-α 而非 facebook/DiT

facebook/DiT 是 class-conditional ImageNet 模型，不支持 text-to-image。PixArt-α 在 DiT 基础上加入了 T5 text encoder + cross-attention，diffusers 原生支持，且有 DreamBooth LoRA 训练生态。

### 决策 2: 为什么弃用自定义 denoising loop

初版 `dit_story_joint.py` 手写了完整的 denoising loop（调度器、CFG、VAE decode）。但 PixArt-α 的 resolution binning (512→1024 映射) 和 adaLN conditioning 细节复杂，手写容易出现不可见 bug。改用 `pipeline.__call__(prompt=[...])` 后，所有 pipeline 内部逻辑由 diffusers 保证正确性。

### 决策 3: 为什么从空间对齐改为全局上下文

空间位置对齐的 blending 假设"同位置 token 属于同语义"，这对静态场景有效但对移动角色有害。全局上下文只加全局偏置，不改变空间结构，是保护角色身份的妥协方案。

### 决策 4: 为什么 merge_and_unload 而非保持 PeftModel wrapper

`merge_and_unload()` 将 LoRA 权重 bake 进基座模型，随后 CrossSceneSelfAttentionWrapper 可以正常注入到 transformer blocks。如果保持 PeftModel wrapper，后续访问 `block.attn1` 可能拿到的是 wrapper 而非实际 attention 模块。

### 决策 5: 为什么 LoRA 必须在 CPU offload 之前加载

`enable_model_cpu_offload()` 在 pipeline 各组件上注册 accelerate hook。`PeftModel.from_pretrained()` 包装 transformer，`merge_and_unload()` 解包。如果 offload hook 在 PeftModel wrapper 上，解包后 hook 丢失引用导致 `_hf_hook` AttributeError。

### 决策 6: 为什么 per-layer blend_strength 而非全局统一

浅层特征 (纹理/颜色) 对场景一致性贡献大且不伤角色，可以强融合；深层特征 (语义/身份) 对角色至关重要，应该弱融合或关闭。分层控制比全局 blend_strength 更精细。

---

## 附录: 代码调用链

```
cli.py
  └─ pipeline.run_pipeline(config)
       ├─ parser.parse_story_file(input)           → Story
       ├─ prompt_pipelines.build_prompt_pipeline() → PromptPipeline
       ├─ prompt_pipeline.build(story)             → PromptBundle
       ├─ generators.factory.build_generation_backend()
       │    └─ if backend="dit_story_joint" → DitStoryJointGenerator
       └─ generator.generate_story(request)
            ├─ _inject_cross_scene_attention(N)
            │    └─ transformer_blocks[i].attn1 = CrossSceneSelfAttentionWrapper(...)
            ├─ _update_num_scenes(N)
            └─ pipeline(prompt=[p1, p2, ...], ...)
                 ├─ T5 Encoder: prompts → embeddings
                 ├─ Denoising loop (each step):
                 │    └─ transformer(latents, text_embeds, timestep)
                 │         └─ for each block:
                 │              └─ block.attn1 = CrossSceneSelfAttentionWrapper.forward()
                 │                   └─ global context blending → self-attention
                 ├─ VAE Decoder: per-scene decode
                 └─ ImageProcessor: postprocess → PIL images
```
