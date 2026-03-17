---
name: kibble-case-organize
description: |
  将迁移/适配项目整理为 kibble 案例仓库规范结构，包括 case.toml 元数据、
  deploy 部署产物、devlog 开发日志、showcase 展示材料的完整组织流程，
  包含对话记录导出与摘要总结
triggers:
  - 整理 kibble 案例
  - 按 kibble 规范组织
  - 案例归档
  - 整理项目到 kibble
  - kibble case
  - 案例沉淀
  - 导出对话记录
  - chatlog 整理
---
upstream: [pytorch-npu-migrate, hardware-comparison-report]
downstream: []

# 整理 kibble 案例

## 概述

将已完成的迁移/适配项目，按 kibble 案例仓库规范整理为可归档、可交付的标准结构。

## 前置条件

- kibble 仓库位于 `/data/kibble`
- 规范文档: `/data/kibble/README.md`、`/data/kibble/docs/CONTRIBUTING.md`、`/data/kibble/docs/taxonomy.md`
- 模板: `/data/kibble/templates/case.toml`
- 已有案例可参考: `/data/kibble/cases/*/`

## 工作流程

### 阶段 1: 分析源项目

1. 阅读源项目 README、迁移报告，了解项目背景
2. 盘点现有产物（代码、测试、文档、脚本）
3. 确定 case.toml 分类字段值（参考 `taxonomy.md`）

### 阶段 2: 创建目标结构

目标结构（**必须严格遵守**）:

```
cases/<case-name>/
├── case.toml                  # 元数据（必填，从模板生成）
├── deploy/                    # 部署产物
│   ├── README.md              # 顶层索引，指向项目子目录
│   └── <project-name>/        # 完整可部署项目（不散落在 deploy 根下）
│       ├── README.md          # 环境准备、运行命令、性能数据
│       ├── src/               # 源代码
│       ├── tests/             # 测试
│       ├── scripts/           # 验证脚本
│       ├── models/            # 模型权重（含下载说明）
│       └── ...
├── devlog/                    # 开发日志
│   └── kernelcat-feedback.md  # KernelCAT 能力评估
└── showcase/                  # 展示材料
    ├── summary.md             # 案例总结（面向商务/市场）
    └── chatlog/               # 人机协作日志
        ├── <case-name>.txt    # 完整对话记录（工具导出）
        └── <case-name>-summary.txt  # 对话摘要（去噪提炼）
```

**关键原则**:
- `deploy/` 根下只放 `README.md` 索引 + 项目子目录，**不散落项目文件**
- `deploy/<project-name>/` 是完整可部署项目，用户 `cd` 进去即可运行
- 清理 `.git`、`__pycache__`、`.egg-info` 等非交付物
- `models/checkpoints/` 不含实际权重，放 `README.md` 写下载命令

### 阶段 3: 填写 case.toml

从 `/data/kibble/templates/case.toml` 复制模板，逐节填写:

1. **基础信息**: name, summary, status, created, contacts
2. **分类**: entry_layer, primary_layers, touched_layers, task, scope, tags
   - 关键字段加行内注释说明选择理由
3. **技术栈**: languages, tools, dev_mode
4. **来源与目标**: source/target 的 platform 和 hardware
5. **难度**: level + factors 列表
6. **工作量**: kernelcat 耗时 vs 工程师耗时
7. **KernelCAT 能力**: version, completion, mode, strengths, limitations
8. **环境要求**: hardware, software
9. **产物**: deploy, devlog, showcase 路径
10. **关联信息**: source_repo, docs, paper

字段取值参考: `/data/kibble/docs/taxonomy.md`

### 阶段 4: 编写 devlog/kernelcat-feedback.md

使用 `templates/kernelcat-feedback.md` 模板，包含:

1. **案例概述**: 5 条 bullet（任务、难度、工作模式、完成度、总耗时）
2. **过程记录**: 按阶段记录，每阶段含「KernelCAT 行为」「问题」「人工介入」「结果」
3. **能力评估**: 表格形式，含评分和说明
4. **具体问题记录**: 每个问题含现象、原因、解决、建议
5. **总结**: 整体评价 + 工作量对比表 + 改进建议优先级

**要求**: 如实记录，包括 KernelCAT 做不到的部分

### 阶段 5: 编写 showcase/summary.md

使用 `templates/summary.md` 模板，面向非技术人员:

1. **一句话总结**: 突出核心亮点数字
2. **案例亮点表**: 模型、改动量、耗时、通过率等关键数据
3. **迁移效果**: 用 ✅ 列表展示能力
4. **性能数据**: 简洁的对比表
5. **KernelCAT 能力展示**: 5-6 条能力点
6. **适用场景**: 商务可直接引用的场景描述
7. **相关链接**: 原始仓库、论文等

### 阶段 6: 导出与摘要对话记录 (chatlog)

将本案例的人机协作对话历史导出到 `showcase/chatlog/`，并生成摘要。

#### 背景: 数据源

kcat 每次会话会在 `~/.kerminal/sessions/<year>/<month>/<day>/` 下生成
`rollout-<timestamp>-<uuid>.jsonl` 文件。每个 `.jsonl` 文件记录一次完整
对话的所有消息（用户输入、助手回复、工具调用等），并包含当时的工作
目录 (cwd)，这就是按项目路径过滤的依据。

#### 6.1 发现项目路径

kibble case name 与 kcat 会话中的项目路径不一定相同（如 case 叫 `bioemu`，
但工作目录是 `/data/models/bioemu-test`）。需先查看所有项目列表建立映射：

```bash
python3 /tmp/history-viewer/view_chat_history.py \
  --cli-name kcat --list-projects /root/.kerminal/sessions/
```

输出会列出所有项目路径及其会话数，从中找到与目标 case 对应的路径。
若找不到，说明该案例在其他环境中执行，跳过即可。

#### 6.2 导出完整对话记录

使用 [claude-code-history-viewer](https://github.com/Zhiwen-Liu/claude-code-history-viewer) 工具：

```bash
# 克隆工具（首次使用，已有则跳过）
git clone https://github.com/Zhiwen-Liu/claude-code-history-viewer /tmp/history-viewer

# 导出（用上一步发现的实际项目路径）
python3 /tmp/history-viewer/view_chat_history.py \
  --cli-name kcat \
  --project /data/models/<project-path> \
  --no-thinking \
  --export /data/kibble/cases/<case-name>/showcase/chatlog/<case-name>.txt \
  /root/.kerminal/sessions/
```

工具会扫描 sessions 目录下所有 `.jsonl`，筛出 cwd 匹配 `--project` 的会话，
按时间线拼接、去重后输出为纯文本。

**注意**:
- 导出前先检查 `chatlog/` 目录是否已有记录，已有则跳过
- 一个项目可能有多个会话（不同日期/不同上下文窗口），工具会自动合并

#### 6.3 编写对话摘要

阅读完整对话记录，生成 `<case-name>-summary.txt`，要求：

1. **去噪**: 去除无实质内容的消息（“继续任务”、environment_context、AGENTS.md 注入等）
2. **分阶段组织**: 按任务阶段分组（迁移启动→功能测试→精度对比→项目整理→归档）
3. **保留关键对话**: 每个阶段保留用户的关键指令和助手的核心行动/结果
4. **连贯性**: 确保摘要能完整反映任务从启动到完成的全过程
5. **结果总结**: 末尾附加关键成果汇总（代码改动、测试通过率、性能数据、耗时等）

摘要模板见: `templates/chatlog-summary.txt`

### 阶段 0: 判断创建还是更新

检查 `/data/kibble/cases/` 下是否已有同名 case。若已存在，执行增量更新流程：

1. diff 所有关键文件（源码、文档、脚本）找出不同步的部分
2. 同步变更的文件到 deploy/
3. 更新 case.toml 的 `updated` 日期、`scope`、`strengths`、`limitations` 等字段
4. devlog 追加新阶段（而非重写），更新工作量表
5. showcase/summary 更新性能数据
6. chatlog 摘要追加新轮工作
7. 最后运行 validate-case.py 确认一致性

### 阶段 7: 验证

检查清单:
- [ ] `case.toml` 所有必填字段已填写，取值符合 taxonomy.md
- [ ] `deploy/` 根下只有 README.md + 项目子目录（无散落文件）
- [ ] `deploy/<project>/` 可独立部署（含 README、源码、测试、脚本）
- [ ] `deploy/<project>/models/` 含权重下载说明（不含实际权重文件）
- [ ] 项目目录和工作目录中无冗余权重副本（权重仅存在于缓存目录如 ~/.boltz/)
- [ ] `devlog/kernelcat-feedback.md` 格式正确、内容完整
- [ ] `showcase/summary.md` 面向非技术人员、数据准确
- [ ] `showcase/chatlog/` 目录存在，含完整对话记录和摘要（若本环境有会话历史）
- [ ] 对话摘要按阶段组织、去除噪音、保留关键指令和结果
- [ ] 无 `.git`、`__pycache__`、`.egg-info` 等非交付物
- [ ] 无敏感信息（密码、token 等）

## 参考

- 完整案例: `references/bioemu-case.md`
- 更新已有案例: `references/boltz2-case.md`
- chatlog 导出工具: https://github.com/Zhiwen-Liu/claude-code-history-viewer
- devlog 模板: `templates/kernelcat-feedback.md`
- summary 模板: `templates/summary.md`
- deploy README 模板: `templates/deploy-readme.md`
- chatlog 摘要模板: `templates/chatlog-summary.txt`
