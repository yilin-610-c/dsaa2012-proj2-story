# Task 1: Story (Multi-Shot / Multi-Panel Generation) Requirements and Details

## 1. Task Overview (Goal)
* The core goal is to generate a sequence of images corresponding to a multi-panel text description (shots/panels).

## 2. Output Quality Requirements (Satisfaction)
The generated image sequence must satisfy the following three core conditions:
* **Per-panel correctness**: Each image must accurately match its corresponding panel description.
* **Cross-panel consistency**: Recurring characters, objects, backgrounds, and the overall style must remain consistent across the entire sequence.
* **Narrative continuity**: The story must progress naturally from one panel to the next.

## 3. Output Format Requirements
* You must generate the required number of images for each test case.
* You must strictly follow the official file naming and resolution requirements.

## 4. Evaluation Focus
The system will be evaluated primarily on the hidden holdout test set (Test-B). The evaluation criteria include:
* Prompt adherence for each panel
* Narrative continuity
* Cross-panel consistency
* Style consistency
* Image quality
* **Grading Method**: Grading is based mainly on human evaluation by the TAs and the instructor. It focuses on the relative quality of your submission compared to other teams under the same task setting, rather than an absolute automatic score.
* A strong submission should produce outputs that are faithful to the input, consistent across images, and visually convincing.

## 5. Strict Constraints and General Rules (Important Rules)
* **Data Collection**: Each team is responsible for finding, selecting, and preparing its own training/finetuning data. Any external resources (public datasets, pretrained models, etc.) must comply with the course policy and be fully disclosed in your technical report.
* **Automated Pipeline**: Your submission must be a fixed, automatic, and reproducible pipeline.
* **Prohibited Actions**: 
    * No manual per-case editing is allowed.
    * No hard-coding outputs for specific test cases.
    * No agent-based systems and no external API usage are allowed.