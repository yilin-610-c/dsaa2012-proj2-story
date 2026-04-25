# StoryDiffusion 扩展说明

本文档记录 `dsaa2012-proj2-story` 中我新实现的内容，主要包括：

1. `llm` 解析 `txt` 的结构化提示链路
2. 角色 `anchor` 生成与选择
3. `storydiffusion_direct` 故事级后端
4. 两条可切换的生成管线：
   - **美学优先 pipeline**
   - **身份优先 pipeline**
  （还没完全验证可行性，目前是不行）
5. `StoryDiffusion` 的 consistent attention 原理说明

---

## 1. 现在这套系统在做什么

整个仓库现在已经不只是一个“单纯文生图”的 baseline，而是把故事生成拆成了几层：

- **文本理解层**
  - 读取 `test_set/*.txt`
  - 解析 scene block
  - 提取人物、场景、动作、代词等信息

- **角色规划层**
  - 生成 `character_specs`
  - 识别主角、性别、职业等结构化信息
  - 为后续 anchor 生成准备角色描述

- **身份资产层**
  - 为每个角色生成 portrait / half-body anchor
  - 自动从 3 张 half-body 候选中选出 canonical 图

- **故事后端层**
  - `storydiffusion_direct` 接收整条 story 的 `scene_plans`
  - 按 scene 调度生成
  - 可将 anchor 作为 reference image 注入

- **输出层**
  - 保存 `story_scene_plans.json`
  - 保存 `story_backend_request.json`
  - 保存 scene prompt / scene result / selected image

---

## 2. 新增的核心能力

### 2.1 LLM / Rule-based prompt pipeline

现在可以先把 `txt` 解析成结构化故事，再由 prompt pipeline 产出：

- `PromptSpec`
- `PromptBundle`
- `character_specs`
- `scene_route_hints`

LLM 失败时会自动 fallback 到 rule-based，不会直接断掉整条生成链路。

### 2.2 Anchor Bank

Anchor Bank 会为每个角色生成：

- `portrait.png`
- `half_body_cand_0.png`
- `half_body_cand_1.png`
- `half_body_cand_2.png`
- `canonical_half_body.png`
- `canonical_anchor.json`

其中 `canonical_half_body.png` 是后续 identity conditioning 默认使用的 anchor。

### 2.3 `storydiffusion_direct`

`storydiffusion_direct` 不再只是空壳，它现在能：

- 接收 `StoryGenerationRequest`
- 消费 `scene_plans`
- 为每个 scene 构造 `GenerationRequest`
- 将 anchor 路径作为 `reference_image_path` 传给底层生成器
- 生成 scene 结果并落盘

### 2.4 Consistent Attention 独立模块

我把 StoryDiffusion 相关的 consistent attention 原理抽成了独立文件：

- `src/storygen/generators/consistent_attention.py`

它用于后续把原始 Gradio 里的“共享身份记忆”机制接到正式工程里。
（还没有验证过）
---

## 3. 两条 pipeline 的设计

为了平衡“画面好看”与“人物一致性”，现在思路上分成两条管线：

### 3.1 美学优先 pipeline

目标：
- 保留构图自由度
- 保留场景感和镜头感
- 不强制依赖 anchor 图

适合：
- 主输出
- 对美感要求更高的生成

特点：
- 不强挂 anchor
- 降低 identity conditioning 强度
- 让 SDXL 更多按 prompt 自由发挥

### 3.2 身份优先 pipeline

目标：
- 强化人物一致性
- 用于调试和身份锁定
- 便于检查角色是否稳定

适合：
- 测试 anchor 是否生效
- 验证同一角色的脸、发型、衣着是否稳定

特点：
- 强 reference
- 强 IP-Adapter conditioning
- 更容易出现近景、大头图、构图保守

---

## 4. 为什么会出现“anchor 太强，画面变窄”

这是一个非常典型的 trade-off：

- **anchor / identity 太弱**
  - 画面更自由
  - 场景更丰富
  - 但人物一致性不稳

- **anchor / identity 太强**
  - 人物更像同一个人
  - 但更容易把画面压成半身 / 大头 / 近景
  - 场景和动作空间被吃掉

因此，现在不建议所有主输出都默认使用强身份约束。

更合理的方式是：

- 主输出走 **美学优先 pipeline**
- 身份验证走 **身份优先 pipeline**

---

## 5. 关键源码说明

### 5.1 `src/storygen/pipeline.py`

这里是整条链路的总调度器，负责：

- 解析故事
- 构建 prompt pipeline
- 运行 anchor bank
- 生成 `scene_plans`
- 构建 `StoryGenerationRequest`
- 调用 `storydiffusion_direct`
- 保存结果和日志

### 5.2 `src/storygen/generators/storydiffusion_direct.py`

这里是故事级后端入口。它会：

- 读取每个 scene 的 `identity_plan`
- 判断是否启用 `reference_image_path`
- 调用 scene-level diffusers 生成器
- 逐 scene 收集结果

### 5.3 `src/storygen/generators/diffusers_text2img.py`

这是底层图像生成器，负责：

- 加载 diffusers pipeline
- text2img / img2img 生成
- 如果启用 IP-Adapter，就加载 adapter 并把 anchor 图传入

### 5.4 `src/storygen/generators/consistent_attention.py`

这是 StoryDiffusion 风格的 consistent attention 独立实现，核心思想是：

- 写入身份 token 到 `id_bank`
- 生成后续 scene 时读取这些 token
- 让多个 panel 共享身份记忆

---

## 6. 现在可以怎么跑

### 6.1 主输出：美学优先

建议用于主输出，尽量保留构图和场景感：

```bash
cd /home/xyz/Desktop/xluo/StoryDiffusion/dsaa2012-proj2-story
source .venv/bin/activate
conda run -n ipadapter env PYTHONPATH=src python -m storygen.cli \
  --profile cloud_storydiffusion_debug \
  --input test_set/17.txt \
  --run-name verify_storydiffusion_debug_17_v8
```

如果你希望更偏审美、少 anchor 约束，可以后续给这个 profile 做成更轻的 identity 配置。

### 6.2 身份优先：调试 anchor 是否真的生效

用于检查角色是否稳定、anchor 是否传进去了：

```bash
cd /home/xyz/Desktop/xluo/StoryDiffusion/dsaa2012-proj2-story
source .venv/bin/activate
conda run -n ipadapter env PYTHONPATH=src python -m storygen.cli \
  --profile cloud_storydiffusion_debug \
  --input test_set/17.txt \
  --run-name verify_storydiffusion_debug_identity_17
```

---

## 7. 推荐查看的输出文件

每次运行后，重点看这些文件：

```text
outputs/<run_name>/logs/story_scene_plans.json
outputs/<run_name>/logs/story_backend_request.json
outputs/<run_name>/logs/anchor_bank.json
outputs/<run_name>/logs/prompt_bundle.json
outputs/<run_name>/scenes/scene_001/prompt.json
outputs/<run_name>/scenes/scene_001/selected.png
```

如果你想检查某个角色的 anchor 是否选对了，再看：

```text
outputs/<run_name>/anchors/<character_id>/canonical_anchor.json
```

---

## 8. 目前的使用建议

### 主线建议

如果你追求“好看”，就不要把 anchor 压得太重。建议：

- 主输出使用较弱的 identity conditioning
- 让 prompt 和场景本身主导画面
- 只在必要时用强 reference

### 调试建议

如果你追求“像同一个人”，就切到身份优先模式：

- 用更强的 anchor
- 更高的 IP-Adapter 影响
- 更严格地检查 `canonical_anchor.json`

---

## 9. 备注

目前这套实现的定位是：

- 先把 dsaa 的故事理解、角色规划、anchor 生成接好
- 再逐步把 StoryDiffusion 的 consistent attention 接入正式后端
- 在“好看”和“像”之间保留可切换的空间

如果后续要继续扩展，建议优先把：

1. 美学优先 pipeline 做成默认主输出
2. 身份优先 pipeline 做成调试/校验入口
3. consistent attention 只在身份优先模式里启用或降低强度
