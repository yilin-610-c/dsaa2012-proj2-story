# DiT Pipeline 开发状态记录

> 最后更新: 2026-05-15 | 分支: `integrate/prompt-audit-into-storydiffusion`

---

## 目录

1. [项目概述](#1-项目概述)
2. [已完成：功能清单](#2-已完成功能清单)
3. [已完成：新增/修改文件](#3-已完成新增修改文件)
4. [可正常工作](#4-可正常工作)
5. [已知 Bug / 未修复](#5-已知-bug--未修复)
6. [已尝试但失败 / 效果差的方案](#6-已尝试但失败--效果差的方案)
7. [当前最优工作流](#7-当前最优工作流)
8. [待办](#8-待办)

---

## 1. 项目概述

目标：将 StoryDiffusion 的生图模型从 SDXL U-Net 换成 DiT (PixArt-α) 架构，并引入跨场景一致性和角色一致性机制。

核心改动：
- 新增 2 个 DiT Generator (scene-level + story-level)
- 新增 CrossSceneAttention 跨场景融合机制
- 新增 DreamBooth LoRA 角色一致性训练闭环
- 新增一键 pipeline 脚本

---

## 2. 已完成：功能清单

| # | 功能 | 状态 | 文件 |
|---|------|:---:|------|
| 1 | **dit_text2img** — Scene-level PixArt-α generator | ✅ | `src/storygen/generators/dit_text2img.py` |
| 2 | **dit_story_joint** — Story-level PixArt-α generator | ✅ | `src/storygen/generators/dit_story_joint.py` |
| 3 | **CrossSceneSelfAttentionWrapper** — 跨场景全局上下文融合 | ✅ | `dit_story_joint.py` (class) |
| 4 | per-layer **blend_strength** 分层控制 | ✅ | `dit_story_joint.py` + `configs/base.yaml` |
| 5 | **LoRA 加载** (merge_and_unload) | ✅ | `dit_text2img.py`, `dit_story_joint.py` |
| 6 | **lora_trigger** 注入 prompt | ✅ | `dit_text2img.py`, `dit_story_joint.py` |
| 7 | **Factory 注册** 两个新 backend | ✅ | `src/storygen/generators/factory.py` |
| 8 | **10+ 个新 Config Profile** | ✅ | `configs/base.yaml` |
| 9 | **gen_lora_ref_images.py** — 训练参考图（默认 PixArt-α；可选 SDXL） | ✅ | `scripts/gen_lora_ref_images.py` |
| 10 | **train_pixart_lora_hf.py** — LoRA 训练脚本 | ✅ | `scripts/train_pixart_lora_hf.py` (官方 + 修改) |
| 11 | **run_full_lora_pipeline.sh** — 一键全流程 | ✅ | `scripts/run_full_lora_pipeline.sh` |
| 12 | **dit_pipeline_complete_guide.md** — 使用指南 | ✅ | `docs/dit_pipeline_complete_guide.md` |
| 13 | **dit_full_implementation_report.md** — 实现报告 | ✅ | `docs/dit_full_implementation_report.md` |
| 14 | facebookresearch/DiT 子模块 | ✅ | `.gitmodules` |

---

## 3. 已完成：新增/修改文件

### 新增文件 (项目贡献)

```
src/storygen/generators/dit_text2img.py          ← Scene-level PixArt + LoRA (~130行)
src/storygen/generators/dit_story_joint.py        ← Story-level PixArt + CrossSceneAttention + LoRA (~380行)
scripts/train_pixart_lora_hf.py                   ← PixArt DreamBooth LoRA 训练 (官方 1042行 + 修改)
scripts/gen_lora_ref_images.py                    ← 默认 PixArt 参考图；`--ref-backend sdxl` 为 SDXL+IP-Adapter
scripts/run_full_lora_pipeline.sh                 ← 一键全流程脚本 (~120行)
docs/dit_pipeline_complete_guide.md               ← 使用指南
docs/dit_full_implementation_report.md            ← 实现报告
docs/STATUS.md                                    ← 本文档
```

### 修改文件

```
src/storygen/generators/factory.py                ← 注册 2 个新 backend
configs/base.yaml                                 ← +12 个 profile, +新配置项
.gitmodules                                       ← facebookresearch/DiT 子模块
```

### 新增依赖

```bash
pip install peft tiktoken sentencepiece protobuf datasets torchvision
```

---

## 4. 可正常工作

### 4.1 Scene-level PixArt 生成 ⭐ (推荐)

```bash
PYTHONPATH=src python -m storygen.cli --profile dit_smoke_test --input test_set/01.txt
```

生成质量正常，每场景 ~5-6 秒。

### 4.2 Scene-level PixArt + LoRA ⭐⭐ (当前最优)

```bash
PYTHONPATH=src python -m storygen.cli \
  --profile dit_lora_smoke \
  --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/character/checkpoint-800 \
  --set model.lora_trigger="sks "
```

LoRA 可正常加载和推理。CFG=2.5, 12 steps, rank=32。

### 4.3 LoRA 训练参考图生成 (v4)

```bash
PYTHONPATH=src python scripts/gen_lora_ref_images.py --input test_set/01.txt --output-dir training_data/xxx [--ref-backend pixart|sdxl]
```

Phase 1: anchor_bank + IP-Adapter → canonical character image
Phase 2: canonical image as IP-Adapter ref → 15 diverse images (不同姿势/角度/背景)

### 4.4 LoRA 训练

```bash
/home/xyz/.conda/envs/ipadapter/bin/python -m accelerate.commands.launch \
  scripts/train_pixart_lora_hf.py ... --resolution=512 --rank=32 ...
```

可正常训练，checkpoint 正常保存。**已知**：验证阶段因无 `--validation_prompt` 而崩溃，但不影响 checkpoint 保存（无害错误）。

### 4.5 一键全流程

```bash
bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt
```

三步连贯运行，输出到 `outputs/dit_lora_smoke_full_<time>_<story>/`。Step 1 默认 **PixArt** 产训练图（`--ref-backend sdxl` 恢复旧 SDXL+IP-Adapter）。一键脚本推理时优先加载 `lora_checkpoints/` 根目录的最终 PEFT adapter（避免仅用 `checkpoint-*` 时与训练步数不同步导致人物不一致）。

### 4.6 Factory 注册

`dit_text2img` + `dit_story_joint` 两个 backend 可正常通过 `build_generation_backend()` 创建。

### 4.7 Config Profiles

所有 10+ 个新增 profile (`dit_smoke_test`, `dit_lora_smoke`, `dit_story_joint_smoke` 等) 均可通过 `--profile` 正常加载。

---

## 5. 已知 Bug / 未修复

### 5.1 Story-Joint 跨场景融合效果不如 Scene-Level ❌

**严重程度**: 高

**现象**: `dit_story_joint` 生成质量比 `dit_text2img` (scene-level) 差，色彩方差更大、细节更少。

**根因分析**: 跨场景全局上下文 blending 在去噪早期步骤注入噪声（跨场景均值 `mean(x[all scenes])` 是噪声信号的混合）。Transformer 特征不可简单算术平均。

**详情**: 见 [dit_pipeline_complete_guide.md §5.2](dit_pipeline_complete_guide.md#52-为什么-scene-level-可能比-story-joint-好)

**建议**: 目前不建议使用 story-joint。CrossSceneAttention 需要重新设计。

### 5.2 LoRA 生成结果与训练参考图不够像 ⚠️ (核心问题已定位)

**严重程度**: 高（根因已定位，修复已实施，待重训练验证）

**现象**: 用 checkpoint LoRA 推理，三个 scene 生成的分别是白人男性、黑人男性、白人女性 — 人物身份完全随机。

**根因分析**:
1. ~~T5 text encoder 被冻结~~ → **已修复**: 加入 `--train_text_encoder`，T5 LoRA rank=4
2. **训练数据身份不一致（核心根因）**: Phase 2 用 PixArt text2img 独立生成 15 张图，**每张是不同的人**。16 张图标注为 "sks Ryan" 但实际是 16 个随机人物 → LoRA 学到模糊噪声而非具体身份
3. ~~LoRA 只训练 transformer~~ → **已修复**: 同时训练 transformer LoRA + T5 LoRA
4. **推理 prompt 格式混乱**: `generation_prompt` 包含大量结构性 token，训练 caption 从未出现 → **已修复**: 推理改用 `scoring_prompt` + `style_prompt`
5. **LoRA alpha 偏弱** → **已修复**: `--lora_alpha` 默认等于 `--rank`

**已修复（代码）**:
- `gen_lora_ref_images.py`: Phase 2 **不再做 text2img 随机生人**，改为对 canonical 做图像增强（翻转、裁剪、亮度）→ 12 张图全是同一个人，纯 PixArt 风格，无跨模型污染
- `train_pixart_lora_hf.py:519`: T5 LoRA 保持 fp32（GradScaler 兼容）
- `train_pixart_lora_hf.py:871`: T5 输出显式 cast 到 weight_dtype（避免 fp32→fp16 崩溃）
- `run_full_lora_pipeline.sh`: 默认 `--train_text_encoder`
- `dit_text2img.py` + `dit_story_joint.py`: 推理 prompt 改用 `scoring_prompt` + `style_prompt`

**待验证**: 重新跑 `run_full_lora_pipeline.sh`，确认训练图人物一致 + 推理一致性显著改善。

### 5.3 --train_text_encoder 在 mixed_precision=fp16 下报错 ✅ (已修复)

**严重程度**: 已解决

**现象**: `ValueError: Attempting to unscale FP16 gradients.`

**根因**: 此前 `text_encoder.to(dtype=weight_dtype)` 把 T5 LoRA 参数 cast 到 fp16，PyTorch GradScaler 无法 unscale fp16 梯度。

**修复**: 去掉 `dtype=weight_dtype`，T5 LoRA 参数保持默认 fp32。Base T5 weights 仍为 fp16（冻结不做训练），显存影响可忽略。

### 5.4 训练脚本验证阶段崩溃 ❌ (无害)

**严重程度**: 低 (不影响功能)

**现象**: `ValueError: Provide either 'prompt' or 'prompt_embeds'. Cannot leave both undefined.`

**根因**: 训练脚本在 checkpoint 后尝试验证生成，但 `--validation_prompt` 未设置。

**影响**: checkpoint 正常保存，仅验证图不生成。`run_full_lora_pipeline.sh` 用 `|| true` 吞掉了这个错误。

### 5.5 SDXL-Turbo 直接生成 1024² 图像会变形 ❌

**严重程度**: 低 (已在 v3/v4 中规避)

**现象**: CFG=0 时 SDXL-Turbo @ 1024² × 4 steps 生成扭曲人脸（"两个头"）。

**规避**: gen_lora_ref_images.py v3/v4 使用 anchor_bank @ 768²，以及 Phase 2 @ 768²。

### 5.6 LoRA + CPU offload 顺序敏感 ❌

**严重程度**: 低 (已修复但需注意)

**现象**: 如果先 `enable_model_cpu_offload()` 再加载 LoRA，会报 `AttributeError: _hf_hook`。

**修复**: `_load_lora()` 在 `enable_model_cpu_offload()` 之前调用。

### 5.7 PixArt-α 不支持 IP-Adapter ❌

**严重程度**: 低 (架构限制)

**现象**: `PixArtAlphaPipeline` 没有 `load_ip_adapter()` / `set_ip_adapter_scale()`。

**根因**: diffusers 未给 PixArt 实现 `IPAdapterMixin`，且社区没有 PixArt 版 IP-Adapter 权重。

**影响**: 不能用 IP-Adapter 直接做 PixArt 推理时的角色锚定，必须走 LoRA 训练闭环。

### 5.8 accelerate 命令用系统 Python 而非 conda ❌

**严重程度**: 低 (有 workaround)

**现象**: `which accelerate` → `~/.local/bin/accelerate` (系统 Python 3.13)。

**规避**: 用完整路径 `/home/xyz/.conda/envs/ipadapter/bin/python -m accelerate.commands.launch`。

---

## 6. 已尝试但失败 / 效果差的方案

| # | 方案 | 结果 | 原因 |
|---|------|------|------|
| 1 | 手写 denoising loop (dit_story_joint v1) | 生成乱码 | resolution binning 处理不正确 |
| 2 | 空间位置对齐跨场景 blending (v1) | 角色被洗掉 | 脸部 token 混入其他场景背景 token |
| 3 | 随机 anchor + SDXL-Turbo 直接生成训练图 (v2) | 男女混合、脸部变形 | CFG=0 @ 1024² 质量差 |
| 4 | anchor_bank only 训练图 (v3) | DreamBooth 几乎学不到差异 | 10 张同姿势同背景 |
| 5 | --train_text_encoder + mixed_precision=fp16 (v5) | **已修复** ✅ | T5 LoRA 保持 fp32 避开 GradScaler 限制 |
| 6 | Story-Joint v2 (全局上下文) | 比 scene-level 更差 | 噪声均值注入破坏 denoising |
| 7 | PixArt Phase 2 纯 text2img 生训练图 (v6) | **已修复** ✅ | 改为 canonical 图像增强，不再随机生人 |

---

## 7. 当前最优工作流

### 一键命令

```bash
bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt
```

### 分步命令（调试用）

```bash
# Step 1: 训练图 (pixart: canonical + 图像增强 → 13 张全同一个人)
PYTHONPATH=src python scripts/gen_lora_ref_images.py --input test_set/01.txt --output-dir training_data/xxx [--ref-backend pixart|sdxl]

# Step 2: 训练 (rank=32, 1200 steps)
/home/xyz/.conda/envs/ipadapter/bin/python -m accelerate.commands.launch \
  /absolute/path/scripts/train_pixart_lora_hf.py \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
  --train_data_dir=/absolute/path/training_data/xxx \
  --output_dir=/absolute/path/lora_checkpoints/xxx \
  --resolution=512 --rank=32 --train_batch_size=1 \
  --learning_rate=1e-06 --max_train_steps=1200 \
  --checkpointing_steps=200 --gradient_checkpointing --mixed_precision=fp16

# Step 3: 推理 (CFG=2.5)
PYTHONPATH=src python -m storygen.cli --profile dit_lora_smoke --input test_set/01.txt \
  --set model.lora_path=lora_checkpoints/xxx/checkpoint-1200 \
  --set model.lora_trigger="sks "
```

---

## 8. 待办

### 高优先级

- [x] **修复 --train_text_encoder mixed precision 冲突** — T5 LoRA 保持 fp32 避开 GradScaler fp16 限制 ✅
- [ ] **重新训练 + 验证 LoRA 角色相似度** — 用修复后的 pipeline 重跑，对比改造前后的人物一致性
- [ ] **重新设计 CrossSceneAttention** — 替代简单全局均值 blending (如 late-step-only, similarity-based)

### 中优先级

- [ ] **1024² 分辨率训练** — 需要更多显存或更激进的 memory saving
- [ ] **增加训练图多样性** — 全身、动态姿势、多光照
- [ ] **多角色 LoRA 支持** — 每个角色独立的 trigger token
- [ ] **验证 PixArt-α 的 DreamBooth 效果** — 与 SDXL+IP-Adapter baseline 定量对比

### 低优先级

- [ ] 训练脚本自动设置 `--validation_prompt` 避免无害崩溃
- [ ] DiT 版 IP-Adapter (研究级, 1-2 周)
- [ ] accelerate PATH 修复 (conda vs system)
