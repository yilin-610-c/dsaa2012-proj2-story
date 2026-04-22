## Introduction / Task Understanding

写清楚你们理解的任务难点：

每格要忠实于 panel
跨格要一致
叙事要接得上
不能手工修图，必须是自动、可复现 pipeline

这些都和作业要求完全对齐。

## System Overview

一张总图 + 一段总述：

输入 story panels
prompt generation
generation backbone
reference/anchor conditioning
scoring/reranking
output packaging

## Prompt Generation Module

这里就是你现在的新重点。
建议写：

为什么 naive prompt 不够
你们如何做 structured extraction
API 在这里扮演什么角色
API 输出受哪些约束
为什么这种设计更适合多-panel story task

## Consistency / Generation Module

写：

角色/道具/风格一致性如何注入
哪些 panel 走 text2img，哪些走 img2img 或其他 route
为什么这样设计

## Controlled Comparisons on Test-A

这里只放最关键的对比，不要铺太多。
比如：

prompt variants comparison
maybe 一个整体系统 comparison

## Qualitative Results

挑 8–12 组最好、最能说明问题的例子。
这也是作业明确要求的。

## Failure Analysis

不要回避失败。
写几类典型失败：

API prompt 改写过度
多角色时身份漂移
动作细节没对上
道具丢失
场景切换时 continuity 断裂

## Data / External Resources / Compliance

这部分必须认真写：

用了哪些模型
用了哪些 API
数据怎么来的
有没有做筛选/清洗
如何保证自动化、无 manual per-case editing、无 hard-coding