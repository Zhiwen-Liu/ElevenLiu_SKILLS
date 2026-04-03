# KerminalSKILLS

昇腾（Ascend）NPU 相关 Agent Skills 合集。

## Skill 工作流 (Workflow DAG)

```
pytorch-npu-migrate          tbe-to-ascendc-rewrite
        │                              │
        └────────┬───────────────┘
                │
    heterogeneous-validation
                │
    hardware-comparison-report
                │
        ┌───────┴───────┐
        │               │
  doc-cross-review  kibble-case-organize
```

## Skills 列表

### 昇腾迁移与开发

| Skill | 说明 |
|-------|------|
| [pytorch-npu-migrate](skills/pytorch-npu-migrate/) | PyTorch 模型迁移到昇腾 NPU 完整流程 |
| [tbe-to-ascendc-rewrite](skills/tbe-to-ascendc-rewrite/) | TBE 算子改写为 AscendC 原生 kernel |

### 验证与报告

| Skill | 说明 |
|-------|------|
| [heterogeneous-validation](skills/heterogeneous-validation/) | 异构平台数值正确性验证方法论 |
| [hardware-comparison-report](skills/hardware-comparison-report/) | CPU vs NPU/GPU 对比验证报告编写 |
| [doc-cross-review](skills/doc-cross-review/) | 硬件迁移文档交叉审查 |

### 项目管理

| Skill | 说明 |
|-------|------|
| [kibble-case-organize](skills/kibble-case-organize/) | 迁移适配项目案例归档整理 |

### Session 管理

| Skill | 说明 |
|-------|------|
| [session-handoff](skills/session-handoff/) | Session 结束前生成交接文档，供下一个 session 无缝续接 |
| [session-retro](skills/session-retro/) | 复盘当前 session，分析卡点与改进机会 |

### 通用工具

| Skill | 说明 |
|-------|------|
| [find-skills](skills/find-skills/) | 发现和安装 Agent Skills |
| [self-improvement](skills/self-improvement/) | 捕获学习、错误和纠正，持续改进 |
| [skill-creator](skills/skill-creator/) | 创建、修改和优化 Skills |

## 重构说明

基于 `skill-creator` 和 `self-improvement` 方法论，对 6 个昇腾相关 skills 进行了系统性重构：

1. **Frontmatter 标准化**: 统一 YAML frontmatter 格式，upstream/downstream 纳入 frontmatter
2. **描述优化**: 中英双语描述，增强触发准确性
3. **触发词扩展**: 每个 skill 添加中英文触发词，覆盖更多用户表达
4. **工作流 DAG**: 明确 skill 间依赖关系，形成完整迁移流水线
5. **结构精简**: 消除冗余内容，问题速查表格化，决策树可视化
6. **工具补全**: 为 heterogeneous-validation 添加 validate_precision.py 脚本
