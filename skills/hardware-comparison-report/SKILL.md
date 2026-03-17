---
name: hardware-comparison-report
description: 硬件迁移项目的 CPU vs NPU/GPU 对比验证报告编写方法论
triggers:
  - CPU 和 NPU 对比
  - 精度验证报告
  - 硬件迁移验证
  - 写对比文档
  - 精度性能报告
---
upstream: [pytorch-npu-migrate]
downstream: [kibble-case-organize, doc-cross-review]

# 硬件对比验证报告编写

## 概述

本技能指导编写硬件迁移项目（如 CPU → NPU/GPU）的对比验证报告，覆盖精度、性能、测试通过率三个维度。

## 核心原则

### 原则 1: 结论前置，细节后置

报告结构严格按以下顺序：

```
测试环境
├─ 第1 精度总结    ← 读者 30s 内获取结论
├─ 第2 性能总结    ← 读者 30s 内获取结论
├─ 第3 精度测试细节 ← 需追溯时查看
├─ 第4 性能测试细节 ← 需追溯时查看
└─ 附录 (环境修复/复现命令)
```

总结表用 ✅/⚠️ 视觉标记，一眼可扫。

### 原则 2: 以测试项为锚点组织精度数据

不按“模块”分类，而是逐测试文件梳理所有数值断言。

**操作步骤：**

```bash
# 第一步: 扫描全部数值断言，建立覆盖清单
grep -rl 'allclose\|isclose\|atol\|wasserstein\|assert.*abs' tests/
```

得到清单后，在报告开头放一张覆盖映射表：

```markdown
| 测试文件 | 数值断言数 | 对应章节 |
|---------|:---------:|:--------:|
| test_models.py | 1 | 第3.1 |
| test_xxx.py    | 5 | 第3.2 |
| ...            | ... | ... |
```

确保每个 allclose/isclose 都有对应的 CPU vs NPU 数据，不遗漏。

### 原则 3: 区分确定性差异和随机性差异

凡涉及随机数的测试项，必须单独验证 RNG 一致性：

```python
# 验证 CPU/NPU 同 seed 是否产生相同随机数
torch.manual_seed(42)
cpu_r = torch.randn(5, device='cpu')
torch.manual_seed(42)
npu_r = torch.randn(5, device='npu:0')
print(f'max_diff={( cpu_r - npu_r.cpu()).abs().max()}')
```

在报告中明确标注差异来源：
- ✅ 确定性计算差异 → float32 浮点误差
- ⚠️ 随机性差异 → RNG 实现不同，预期行为，非精度问题

### 原则 4: 建立分层交叉验证链

脚本验证的基础算子精度与 tests/ 的模型级精度形成信任链：

```
底层: 基础算子 (matmul/softmax/LN) → 保证模型前向精度
中层: SDE/ODE 数值              → 保证采样器分布正确
上层: 单步精度 + 多步累积         → 保证端到端结果正确
```

报告中显式写出这条链，让读者知道各验证项的依赖关系。

### 原则 5: 精度对比表统一 4 列标准格式

| 字段 | 含义 |
|------|------|
| max_abs_diff | 最坏情况 |
| mean_abs_diff | 整体水平 |
| max_rel_diff | 小值放大效应 |
| mean_rel_diff | 相对误差整体水平 |

加上 atol 阈值和 PASS/FAIL 标记。

**atol 设定规则** (必须在文档中明确记录):
- 每个算子的 atol 根据其计算特性单独设定，不使用统一值
- matmul: atol 随内积长度 N 放大，近似 O(N × ε)，如 N=1024 时 atol ≈ 2e-3
- softmax/layernorm: 无大规模累加，atol=1e-5 即可
- gelu/silu: NPU 和 CPU 可能用不同近似实现，atol 需适当放宽 (如 5e-4)
- bf16: 用 ULP 分析确定 atol，见下文

**bf16/fp16 精度的 ULP 分析方法**:
bf16 仅 7 位尾数，对于绝对值为 V 的元素，1 ULP = V × 2^(-7) ≈ V × 0.0078。
例如绝对值 ~16 的元素，1 ULP = 0.125，atol 应设为 ≥ 0.125。
文档中必须解释: "max_abs=0.125 等于该数值范围的 1 ULP，是 bf16 格式的固有精度极限"。

**每个 PASS/DIFF 必须附带解释** (不能只有数据没有解释):
- PASS: 解释为何在阈值内（如 "内积长度 256，累加误差小"）
- max_abs 接近阈值: 解释为何仍符合预期（如 "NPU Cube 分块累加顺序不同"）
- max_rel_diff 异常大: 解释 "小值放大效应"，引导读者看 mean_rel_diff
- DIFF: 解释差异来源和是否可接受

写一个通用 `report()` 函数复用：

```python
def report(label, cpu_val, npu_val, atol=1e-5):
    c = cpu_val.detach().cpu().float().numpy().ravel()
    n = npu_val.detach().cpu().float().numpy().ravel()
    ad = np.abs(c - n)
    rd = ad / np.maximum(np.abs(c), 1e-12)
    ok = np.allclose(c, n, atol=atol, rtol=1e-5)
    tag = "PASS" if ok else "DIFF"
    print(f"  [{tag}] {label}")
    print(f"    max_abs={ad.max():.2e}  mean_abs={ad.mean():.2e}")
    print(f"    max_rel={rd.max():.2e}  mean_rel={rd.mean():.2e}")
    return {"label": label, "max_abs": ad.max(), "ok": ok}
```

### 原则 6: 性能数据必须标注测量条件

每行性能数据必须说明：
- 数据规模 (batch_size, seq_len, 维度)
- 是否含 IO / 后处理
- 是否含 NPU 同步 (`torch.npu.synchronize()`)

反直觉结果（NPU 更慢）必须给出原因分析。

### 原则 7: 复现命令作为文档一部分

附录中放置可直接 copy-paste 执行的命令，不需要额外配置。

## 工作流程

```
1. 扫描 tests/ 的全部数值断言 → 建立覆盖清单
2. 逐文件在 CPU 和 NPU 上运行，记录耗时 + 通过情况
3. 对每个数值断言提取 CPU vs NPU 的 4 列精度数据
4. 编写脚本补充验证基础算子/SDE/多步累积，形成交叉验证
5. 验证 RNG 一致性，对随机性差异定性标注
6. 按模板填充报告: 总结表 → 细节 → 附录
```

## 参考案例

- BioEmu 项目: `references/bioemu-case.md`
- Boltz2 项目: `references/boltz2-case.md`
- 报告模板: `templates/report.md`
- 通用工具函数: `templates/comparison_utils.py`（复制到项目 scripts/ 目录使用）
