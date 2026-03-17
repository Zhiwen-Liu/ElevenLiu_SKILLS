---
name: pytorch-npu-migrate
description: PyTorch 模型迁移到华为昇腾 NPU 的完整流程，包括代码适配、测试验证、精度对齐和性能优化
triggers:
  - PyTorch 模型迁移到 NPU
  - torch 适配昇腾
  - 模型移植到华为 NPU
  - enformer/transformers/CNN 迁移 NPU
---
upstream: []
downstream: [hardware-comparison-report, kibble-case-organize]

# PyTorch 模型迁移到昇腾 NPU

## 概述

本技能指导将 PyTorch 模型迁移到华为昇腾 NPU，基于 `torch_npu + transfer_to_npu` 零代码修改方案。

## 迁移流程

### 1. 环境准备

```bash
# 加载昇腾环境
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<卡号>  # 可选，指定 NPU 卡号

# 检查环境
npu-smi info
python -c "import torch_npu; print(torch_npu.__version__)"
```

> **卡号设置规范**: `ASCEND_RT_VISIBLE_DEVICES` 是可选配置，文档和代码中应标注为"可选，指定 NPU 卡号"。
> Python 代码中使用 `os.environ.setdefault()` 而非直接赋值，避免覆盖外部设置。

### 2. 代码适配模式

根据项目情况选择以下两种方案之一：

#### 方案 A: transfer_to_npu 零代码修改 (推荐首试)

适用：代码中使用 `cuda` 硬编码设备，且不需要同时支持 CPU/CUDA/NPU 多后端。

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # 自动将 cuda 调用转为 npu

# 原始代码无需修改，cuda 调用自动重定向到 NPU
model = Model().cuda()  # 实际分配到 NPU
```

**优点**: 改动量为零，快速验证。
**缺点**: 隐式转换，不支持多后端切换；禁用 `torch.jit.script`。

**与 Lightning Trainer 配合**: 使用 transfer_to_npu 后必须设置 `accelerator="gpu"`（而非 `"cpu"`）。
transfer_to_npu 会将 CUDA 调用重定向到 NPU，Lightning 识别 NPU 为 CUDA 设备后自动管理设备分配。
若使用 `accelerator="cpu"` + 手动 `model.to(npu)`，Lightning 会在 predict/fit 时将模型强制移回 CPU。

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu

trainer = Trainer(accelerator="gpu", devices=1)  # 不要用 "cpu"
trainer.predict(model, datamodule=dm)  # 模型和数据自动在 NPU 上
```

参考案例: Enformer (`references/enformer-case.md`), Boltz2 (`references/boltz2-case.md`)

#### 方案 B: 手动设备抽象层 (更通用)

适用：项目需同时支持 CPU/CUDA/NPU，或原代码不依赖 cuda 硬编码。

```python
# 新建 device.py，透明支持多后端
# 关键: 延迟导入 torch_npu，避免 CPU 模式下初始化 NPU context 占用显存
import os
import torch
from functools import lru_cache

@lru_cache(maxsize=1)
def _check_npu() -> bool:
    if os.environ.get("FORCE_CPU", "") == "1":
        return False
    try:
        import torch_npu  # noqa: F401  # 仅在首次调用时导入
        return torch.npu.is_available()
    except ImportError:
        return False

def get_device(device=None):
    if device:
        return torch.device(device)
    if _check_npu():
        return torch.device('npu:0')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')
```

> **为何延迟导入**: `import torch_npu` 会立即初始化 NPU context 并占用显存。
> 若在模块顶层导入，即使 `--accelerator cpu` 也会占用 NPU 资源。
> 使用 `@lru_cache` 确保只在首次调用时检测，环境变量 `FORCE_CPU=1` 可完全跳过。

然后在模型/采样代码中将 `torch.device('cuda')` 替换为 `get_device()`。

**优点**: 显式控制，支持多后端，可通过环境变量切换。
**缺点**: 需手动修改 2~5 个文件。

参考案例: BioEmu (`references/bioemu-case.md`)

### 3. 测试验证项

| 测试类型 | 验证内容 |
|----------|----------|
| 推理测试 | 前向传播、多头输出、embeddings |
| 训练测试 | 前向+反向、损失收敛、梯度有效性 |
| 精度测试 | CPU vs NPU 余弦相似度 (>0.999) |
| 性能测试 | 推理延迟、吐吐量、显存占用 |

### 4. 常见问题处理

#### 4.1 算子回退 CPU

```
Warning: The operator 'aten::xxx' is not currently supported on the NPU backend
```

**处理**: 通常无需处理，性能影响很小。常见回退算子:
- `aten::lgamma` - 位置编码
- `aten::poisson` - 波松分布采样

#### 4.2 transformers 5.x 兼容性

```python
# 问题: from_pretrained 报错 'all_tied_weights_keys'
# 解决: 手动加载
from huggingface_hub import hf_hub_download
import json

config_path = hf_hub_download(repo_id=model_id, filename='config.json')
model_path = hf_hub_download(repo_id=model_id, filename='pytorch_model.bin')

with open(config_path) as f:
    config = json.load(f)

model = Model.from_hparams(**config)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
```

#### 4.3 DataLoader 多进程 segfault

NPU 环境下 `num_workers > 0` 可能导致 worker 进程 segfault：

```
RuntimeError: DataLoader worker (pid xxx) is killed by signal: Segmentation fault.
```

**原因**: fork 出的子进程继承了 NPU context，但 NPU 驱动不支持跨进程共享。

**处理**: 设置 `num_workers=0`（单进程加载），或使用 `start_method="spawn"`：

```python
dl = DataLoader(dataset, num_workers=2, multiprocessing_context="spawn")
```

#### 4.4 JIT 编译限制

`transfer_to_npu` 禁用 `torch.jit.script`，如需 JIT 需单独处理相关模块。

#### 4.5 设备转移

```python
# 部分函数返回 CPU tensor
one_hot = str_to_one_hot(dna).to(device)  # 手动转移
```

### 5. 优化建议

#### 优化器
```python
from torch_npu.optim import NpuFusedAdamW
optimizer = NpuFusedAdamW(model.parameters(), lr=1e-4)
```

#### 显存优化
```python
# Gradient Checkpointing
config = ModelConfig(use_checkpointing=True)
```

#### 混合精度
```python
from torch.npu.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(x, target=y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 项目结构模板

```
model-npu/
├── model_npu/              # NPU 适配模块
│   ├── __init__.py
│   ├── npu_utils.py        # init_npu, load_pretrained
│   ├── inference.py        # 推理封装
│   └── train.py            # 训练封装
├── data/                   # 测试数据
├── test_npu.py             # 统一测试脚本
├── example.py              # 使用示例
├── docs/API.md
├── README.md
└── MIGRATION_REPORT.md     # 迁移报告
```

## 迁移报告模板

迁移报告应包含:

1. **项目概述** - 源项目、版本、目标平台
2. **环境配置** - 硬件/软件环境、依赖包
3. **迁移方案** - 技术选型、代码修改
4. **功能验证** - 测试概览、详细结果
5. **精度验证** - CPU vs NPU 对比
6. **性能分析** - 推理/训练性能、显存占用
7. **已知问题** - 问题描述和解决方案
8. **文件清单** - 项目文件说明

## 参考案例

- Enformer (方案 A: transfer_to_npu): `references/enformer-case.md`
- BioEmu (方案 B: 手动设备抽象层): `references/bioemu-case.md`
