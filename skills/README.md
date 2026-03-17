# KernelCAT Skills

本目录存放可复用的技能文件。

## 可用技能

| 技能 | 描述 | 路径 |
|------|------|------|
| pytorch-npu-migrate | PyTorch 模型迁移到昇腾 NPU | `pytorch-npu-migrate/SKILL.md` |
| hardware-comparison-report | 硬件迁移的 CPU vs NPU 对比验证报告编写 | `hardware-comparison-report/SKILL.md` |
| kibble-case-organize | 将迁移项目整理为 kibble 案例规范结构 | `kibble-case-organize/SKILL.md` |
| tbe-to-ascendc-rewrite | TBE算子改写为AscendC的系统性方法论(7步流程) | `tbe-to-ascendc-rewrite/SKILL.md` |

## 技能上下游关系

三个技能覆盖了“迁移 → 验证 → 归档”的完整工作流：

```
pytorch-npu-migrate          ──────────────────────────────────────────
  ↓  产出迁移后的代码 + 测试
hardware-comparison-report   ── 产出 CPU vs NPU 对比报告
  ↓  产出验证报告 + 性能数据
kibble-case-organize         ── 产出可归档的标准案例

tbe-to-ascendc-rewrite       ── 独立流程: TBE算子改写为AscendC
  ↓  7步流程: profile -> 参考 -> 架构 -> 实现 -> 调优 -> 测试 -> 工程化
```

单独使用任一技能也可以，不强制要求顺序执行。

## 使用方法

在对话中提及以下关键词时自动触发:

- `$pytorch-npu-migrate`: "PyTorch 迁移 NPU"、"torch 适配昇腾"、"模型移植 NPU"
- `$hardware-comparison-report`: "CPU NPU 对比"、"精度验证报告"、"写对比文档"
- `$kibble-case-organize`: "整理 kibble 案例"、"案例归档"、"整理项目到 kibble"

## 技能结构

```
skills/<skill-name>/
├── SKILL.md           # 技能定义和流程
├── templates/         # 代码/文档模板
└── references/        # 参考案例
```

## 同步说明

本目录 (`/root/.kernelcat/skills/`) 为权威源。`/data/skills/` 为副本，更新后需同步：

```bash
cp -a /root/.kernelcat/skills/* /data/skills/
```
