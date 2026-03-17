# 异构平台模型迁移——数值正确性验证方法论指南
>
> **本文核心问题**: “模型在 NPU 上跑通了，结果也出了，怎么评估结果对不对”

---

## 0. 先说结论：“对不对”这个问题本身就问错了

当你把一个模型从 CPU 搬到 NPU，自然的第一反应是拿 NPU 的输出和 CPU 的输出比——发现不一样——然后纳闷“这到底对不对”。

这个思路的根本问题：**CPU 的结果也不是“真值”**。CPU 的浮点计算同样有舍入、有分块、有累加顺序选择。你拿两个都不精确的东西比差异，得到的只是“两种近似之间的距离”，不是“误差”。

真正该问的问题是：

> **“NPU 上的模型，是否保持了原模型在目标任务上的能力？”**

这不是一个“数字对不对”的问题，而是一个“功能等价性”的问题。数字对比是手段，不是目的。

### 这意味着什么

| 错误思路 | 正确思路 |
|---------|----------|
| “NPU 和 CPU 的 matmul 差了 1e-3，可以接受吗？” | “这个 1e-3 的差异传播到模型终端后，任务指标还在可接受范围内吗？” |
| “allclose 过了，精度没问题” | “allclose 过了只说明数字接近，还需要任务级指标确认功能保持” |
| “CPU 结果是 ground truth” | “CPU 结果是 reference，不是 truth；业务指标才是 truth” |


---

## 1. 验证金字塔：自顶向下定标准，自底向上排故障

这是本指南的核心框架。业界成熟的异构迁移验证（TensorRT、ONNX Runtime、MLC-LLM、OpenVINO）都收敛到了类似的分层结构：

```
+---------------------------------------------------+
|  L0  应用级：任务指标是否等价？                      |  <- "对不对"的最终裁判
+---------------------------------------------------+
|  L1  模型级：输出分布是否等价？                      |  <- 统计等价性
+---------------------------------------------------+
|  L2  模块级：同输入下每个子模块输出是否接近？         |  <- 定位问题模块
+---------------------------------------------------+
|  L3  算子级：单个 op 输出是否接近？                  |  <- 单元测试 / debug
+---------------------------------------------------+

         ^ 定义"对"的方向              ^ debug 的方向
         (自顶向下)                    (自底向上)
```

### 核心原则

1. **L0 是唯一的"对不对"裁判者。** 如果任务指标等价，迁移就是正确的——即使底层算子的数值差异看起来"很大"。
2. **L1-L3 是证据和 debug 工具，不是结论。** 算子级 allclose 通过不等于"迁移正确"，不通过也不等于"迁移失败"。
3. **阈值自顶向下流动。** L0 的任务容忍度决定 L1 的分布差异容忍度，进而决定 L2/L3 的数值阈值。不是反过来。

### 为什么不能只做 L3

这是原报告的核心问题。报告把大量精力花在 L3（单算子 allclose），然后试图用一个 "底层算子精度一致 -> 保证模型前向精度"的逻辑链往上推。但这个推理在数学上不成立：

- 深度网络是数百层非线性变换的复合，每层的微小差异可以被放大、抑制、或以复杂方式交互
- 单算子 diff = 1e-5 不意味着 100 层后的 diff = 100 x 1e-5，它可能是 1e-8（衰减），也可能是 1e-1（放大）
- 没有模型级的实验验证，算子级的结论无法往上传递

### 为什么不能只做 L0

反过来，只做 L0 也不够：

- L0 通过只能说明"在这批测试数据上没问题"，无法保证所有 corner case
- L0 失败时，没有 L1-L3 的数据你根本不知道问题出在哪里
- 只有 L0 的报告无法说服技术评审者，因为缺乏"为什么能工作"的解释

**四层都要做，但各有分工。**

### 每层的角色和工具

| 层级 | 回答的问题 | 典型工具/指标 | 在报告中的作用 |
|------|------------|------------|------------|
| **L0 应用级** | 模型还能用吗？ | benchmark 数据集上的任务指标（RMSD, TM-score, BLEU, mAP...） | **结论** |
| **L1 模型级** | 输出分布一致吗？ | N 次运行的统计分布对比；确定性前向的输出对比 | **主要证据** |
| **L2 模块级** | 哪个子模块引入了差异？ | 逐层 hook 对比、误差传播图 | **辅助证据 / debug** |
| **L3 算子级** | 单个 op 的数值行为是否符合预期？ | allclose、统计量、分布图 | **补充证据 / 单元测试** |

---

## 2. 每层具体怎么做

### L0 应用级：你的"参照"在这里

这是回答"参照什么"的地方。参照不是 CPU 的输出，而是**模型在标准数据集上应有的表现**。

**操作方法：**

1. **找到模型的标准 benchmark。** 几乎所有正经模型都有：
   - 蛋白质结构预测：CASP15、CAMEO -> TM-score, lDDT, GDT-TS
   - LLM：MMBench、HumanEval、MMLU -> accuracy, pass@k
   - CV：ImageNet、COCO -> top-1 acc, mAP
   - 如果找不到公开 benchmark，用模型作者提供的测试集
2. **分别在参考平台和目标平台跑同一个 benchmark。**
3. **比较任务指标，不是比较原始输出。**

以 Boltz2 为例：

```python
# 不要这样（比较原始数字）：
assert np.allclose(npu_confidence, cpu_confidence, atol=0.05)

# 应该这样（比较任务指标）：
# 1. 在 CAMEO 测试集上跑 NPU 和 CPU
# 2. 比较两端的 TM-score 分布
# 3. 确认 NPU 的中位数 TM-score 与 CPU 无统计显著差异
```

**如果 L0 通过，迁移就是正确的。** 即使底层算子的 allclose "看起来"差异很大，只要终端指标 OK，那就是 OK。这就是为什么 L0 是"裁判者"。

**万一真的没有 benchmark 怎么办？**

极少数情况下模型确实没有公开的评测基准。此时的降级策略：

1. **找模型作者要测试用例。** 大多数 repo 的 `examples/` 或 `tests/` 目录有。
2. **自建小型评测集。** 找 20-50 个代表性输入，人工判断输出是否合理。
3. **用多参考平台交叉验证。** 同时在 CPU、CUDA GPU、NPU 上跑，如果 NPU-CPU diff 与 GPU-CPU diff 在同一量级，NPU 就是可信的。

### L1 模型级：统计等价性

L1 的核心是：**比较输出的分布，而不是单个输出的数值**。这对确定性模型和随机模型有不同的做法。

**确定性模型（如 ResNet、BERT 推理）：**

同一输入应产生接近的输出。此时可以逐元素比较：

```python
# 确定性模型：用相同输入，比较输出
cpu_out = model_cpu(input_tensor)
npu_out = model_npu(input_tensor.npu())
diff = (cpu_out - npu_out.cpu()).abs()
# 汇报完整统计量（见第 7 节）
```

但要注意：即使是"确定性"模型，很多 PyTorch op 本身就不是跨平台确定的（如 `torch.nn.functional.interpolate`、某些 conv 实现）。所以即使确定性模型，也应该用多个输入的统计分布来下结论，不要只看一个样本。

**随机模型（如扩散模型、VAE、LLM 采样）：**

单次输出天然不同，比较单次输出无意义。必须比较分布：

```python
# 随机模型：各跑 N 次，比较指标分布
cpu_scores = [run_cpu(input, seed=s) for s in range(N)]
npu_scores = [run_npu(input, seed=s) for s in range(N)]
# Welch t-test 或 Mann-Whitney U 检验分布是否显著不同
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(cpu_scores, npu_scores)
# p > 0.05 -> 无统计显著差异
```

还有一个可以同时做的关键实验：**固定噪声对照（控制变量法）**。这是原报告缺失的最关键实验：

```python
# 在 CPU 上生成噪声，然后分别喝给两端的去噪器
noise = torch.randn(shape, device='cpu')
cpu_result = denoise_cpu(noise)
npu_result = denoise_npu(noise.npu())
# 现在的 diff 纯粹来自计算差异，不含 RNG 影响
```

这个实验能将 RNG 差异和计算差异彻底分离。原报告声称"置信度差异来自 RNG"，但没做这个对照实验，就无法排除"计算差异也有贡献"的可能。

### L2 模块级：定位问题的显微镜

L2 的目的不是"证明正确"，而是当 L0/L1 出现问题时，快速定位到哪个子模块。即使 L0/L1 没问题，L2 数据也能增强报告的说服力（"不仅终端 OK，中间每一步也 OK"）。

具体做法：在模型的关键子模块边界挂 hook，用相同输入跑前向，比较每个子模块的输出。

```python
import torch

def compare_modules(model_cpu, model_npu, sample_input, key_modules):
    """key_modules: ["模块名称列表，如 'encoder', 'decoder.layer.0'"]"""
    hooks_cpu, hooks_npu = {}, {}

    def make_hook(store, name):
        def hook(module, input, output):
            store[name] = output.detach().cpu().float()
        return hook

    for name in key_modules:
        mod_cpu = dict(model_cpu.named_modules())[name]
        mod_npu = dict(model_npu.named_modules())[name]
        mod_cpu.register_forward_hook(make_hook(hooks_cpu, name))
        mod_npu.register_forward_hook(make_hook(hooks_npu, name))

    with torch.no_grad():
        model_cpu(sample_input)
        model_npu(sample_input.npu())

    for name in key_modules:
        diff = (hooks_cpu[name] - hooks_npu[name]).abs()
        print(f"{name}: max={diff.max():.2e}, "
              f"mean={diff.mean():.2e}, "
              f"P99={diff.quantile(0.99):.2e}")
```

看误差是在哪一层突然放大的——那层就是问题所在。

### L3 算子级：单元测试，不是证明

L3 的作用是：
- 确认单个算子的 NPU 实现没有 bug（如输出全零、NaN、形状错误）
- 建立单算子的数值差异 baseline，供 L2 debug 时参考
- **不能用来证明"模型迁移正确"**

#### 选算子

不需要测所有 op，优先覆盖模型中的计算密集型算子和已知有跨平台差异的算子：

| 优先级 | 算子类型 | 原因 |
|--------|----------|------|
| P0 | matmul / bmm | 计算量最大，分块策略差异最显著 |
| P0 | attention (scaled_dot_product) | 组合算子，多种融合实现 |
| P1 | layernorm / rmsnorm | 涉及归约，精度敏感 |
| P1 | softmax | 涉及 exp + 归约 |
| P2 | gelu / silu / swish | 不同平台可能用不同近似 |
| P2 | linear（含 bias） | matmul + add 的组合 |
| P3 | embedding / gather | 通常精确一致，主要查功能正确性 |

#### 测什么

每个算子至少覆盖三个规模（小 / 模型实际规模 / 大），用随机输入：

```python
import torch

def test_op_l3(op_name, op_cpu, op_npu, input_shapes, dtype=torch.float32):
    """L3 单算子对比。op_cpu / op_npu: callable，接受 tensor 返回 tensor"""
    for shape in input_shapes:
        x = torch.randn(shape, dtype=dtype)
        cpu_out = op_cpu(x)
        npu_out = op_npu(x.npu()).cpu()

        # 1. 硬性 sanity check（不过就是 bug）
        assert not torch.isnan(npu_out).any(), f"{op_name} {shape}: NaN!"
        assert not torch.isinf(npu_out).any(), f"{op_name} {shape}: Inf!"
        assert cpu_out.shape == npu_out.shape, f"{op_name} {shape}: shape mismatch!"

        # 2. 输出完整统计量（见第 7 节 report_diff 函数）
        report_diff(f"{op_name} {shape} {dtype}", cpu_out, npu_out)
```

#### 怎么判 PASS/FAIL

L3 有两类判定，性质完全不同：

| 判定类型 | 标准 | 失败含义 |
|----------|------|----------|
| **硬性 (sanity)** | 无 NaN、无 Inf、shape 一致、dtype 一致 | NPU 实现有 bug，必须修 |
| **软性（数值接近度）** | 统计量是否在合理范围 | 需结合 L0-L2 判断是否可接受 |

硬性判定是二值的：过或不过。

软性判定**不应该在 L3 层面下 PASS/FAIL 结论**。L3 的统计量是数据，不是判决。正确的做法是：
- 记录统计量
- 在 L2 层观察这些差异是否被放大
- 在 L0/L1 层确认是否影响任务指标
- 只有当 L0 通过后，才能回过头说"L3 的这些差异是可接受的"

如果你一定要在 L3 层设一个预警线（用于 CI 自动化），使用第 3 节的方法 B（多平台交叉对比）或方法 D（业界惯例）作为依据，并在报告中明确标注"此阈值为预警线，非最终判定"。

**完整的统计量输出格式见第 7 节。**

---

## 3. 阈值怎么定：四种有根据的方法

这是实践中最头疼的问题。先说原则：

> **阈值必须有独立于观测值的依据。**
>
> 如果你的 atol 是看到跑出来的数据后“调”出来的，那它就不是标准，而是循环论证。

以下是四种业界认可的阈值来源，按优先级排序：

### 方法 A：从 L0 任务容忍度反推（最优）

通过扰动实验确定模型对数值差异的实际敏感度：

```python
# 在参考平台上做扰动实验
def perturbation_sensitivity(model, inputs, noise_levels, task_metric_fn):
    """在 CPU/GPU 上给中间层加受控噪声，看任务指标变化多少"""
    baseline = task_metric_fn(model(inputs))
    results = []
    for eps in noise_levels:  # [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        perturbed = add_noise_to_activations(model, inputs, eps)
        metric = task_metric_fn(perturbed)
        results.append((eps, abs(metric - baseline) / baseline))
    return results
    # 找到任务指标变化 <5% 的最大 eps -> 就是你的容忍度
```

比如你发现 eps=1e-3 时 TM-score 变化 <1%，而你的 NPU 逐层 diff 都在 1e-4，那你就有了安全余量。

这个方法的优点：不需要知道任何硬件实现细节，完全从模型行为出发。

### 方法 B：多参考平台交叉比较（推荐）

同时在 CPU、CUDA GPU、NPU 上跑，比较所有两两配对的 diff：

```
GPU-CPU diff: max_abs = 2.3e-4
NPU-CPU diff: max_abs = 1.8e-3
NPU-GPU diff: max_abs = 1.7e-3
```

GPU 是业界已充分验证的平台。如果 NPU-CPU diff 与 GPU-CPU diff 在同一数量级（或 NPU-CPU 仅比 GPU-CPU 大 1 个数量级以内），NPU 就是可信的。

这个方法的优点：提供了一个已知可接受的 diff 参考线，不需要理论推导。

### 方法 C：同平台非确定性基线（实用）

很多平台自身就有跨 run 的非确定性（特别是用了 cuBLAS 的 GPU）：

```python
# 在参考平台上跑多次，看自身的方差
results = [model(input) for _ in range(20)]
variance = torch.stack(results).var(dim=0)
# 如果 NPU-CPU diff 与参考平台自身的 run-to-run variance 在同一量级
# -> diff 属于正常浮点波动
```

### 方法 D：业界惯例参考（作为保底）

当以上方法都不可行时，可以引用业界惯例作为 baseline，但必须明确标注来源：

| 来源 | fp32 典型容忍度 | 说明 |
|------|--------------|------|
| ONNX Runtime | atol=1e-3, rtol=1e-3 | 用于跨后端一致性测试 |
| TensorRT | >99% 元素 rel_diff < 1% | fp32->fp16 的验收标准 |
| PyTorch torch.testing | atol=1e-5, rtol=1.3e-6 | 用于同平台回归测试，跨平台偏严 |
| 实践经验 | 模型输出层 cosine_sim > 0.999 | 用于整体输出相似度判断 |

**重要：业界惯例作为保底，不是标准。** 每个模型的容忍度不同，最理想的方式还是方法 A。

### 原报告的阈值问题（反面案例）

原报告的阈值设定是典型的“后设阈值”（post-hoc threshold）：

```
matmul max_abs = 1.59e-3  -> atol = 2e-3   (余量 1.26x)
gelu  max_abs = 4.73e-4  -> atol = 5e-4   (余量 1.06x)
```

这里的问题不是 atol 本身的值，而是 **atol 的依据缺失**。报告既没有引用业界惯例，也没有做扰动实验，也没有多平台交叉对比。读者无法判断 2e-3 的 atol 到底是"确实安全"还是"刚好能过"。

**正确做法：** 在报告中写清阈值的来源。例如：

> atol=2e-3 依据：扰动实验显示模型在中间层 eps=1e-2 时 TM-score 变化 <2%，
> 当前算子层 diff (1.59e-3) 远小于此容忍度。

或者：

> atol=2e-3 依据：同模型在 CUDA GPU 上的 matmul diff 为 3.2e-4，
> NPU 的 1.59e-3 高于 GPU 约 5 倍，但在 L0 benchmark 中两端 TM-score 无显著差异。

---

## 4. 扩散模型的验证逻辑与确定性模型本质不同

Boltz2 是扩散模型，这类模型的输出是从一个分布中采样得到的。这一点根本性地改变了验证逻辑：

| | 确定性模型 (ResNet, BERT) | 随机模型 (扩散, VAE, LLM 采样) |
|---|---|---|
| 同一输入的输出 | 应该接近 | 天然不同（每次都不同） |
| 比较对象 | 单次输出的数值 | 多次输出的统计分布 |
| “对”的定义 | 输出逻元素接近 | 输出分布统计等价 |
| allclose 是否适用 | 适用于 L1/L2 | **不适用**于终端输出 |

### 扩散模型的正确验证流程

```
步骤 1: 固定噪声对照实验（验证去噪器）
   └─ 在 CPU 上生成噪声 tensor
   └─ 分别喝给 CPU 和 NPU 的去噪器
   └─ 比较输出（此时是确定性比较，allclose 适用）
   └─ 这一步验证的是“神经网络部分的计算一致性”

步骤 2: 分布等价性实验（验证采样质量）
   └─ CPU 和 NPU 各独立跑 N 次（N >= 30，用不同 seed）
   └─ 收集任务指标的分布（如 TM-score, confidence）
   └─ 统计检验两个分布是否等价
   └─ 这一步验证的是“端到端采样质量”

步骤 3: 标准 benchmark（L0 裁判）
   └─ 在标准测试集上比较两端的任务指标
```

### 原报告的问题

原报告在 CLI 对比中只跑了 1 次，用 confidence_score 的差异得出"均在合理范围"。问题：

- **样本量为 1，无法做任何统计推断。** 你无法从 1 个样本知道“0.40 和 0.42 的差异是正常波动还是系统偏差”
- **将差异归因于 RNG，但没做控制实验。** “固定噪声对照”是验证这个声明的唯一方法
- **“均在 Boltz2 典型输出范围内”缺乏引用。** 什么是“典型范围”？依据是什么？

---

## 5. 六个坑：缺乏方法论时的典型症状

以下每个坑都不是孤立的“写作错误”，而是缺乏上述验证框架时的自然后果。

### 坑 1：误用理论误差界

**原报告写法：**
> "理论上界为 O(K x eps)，对于 K=1024，预期误差约 1.2e-4。实测 1.59e-3 略高于理论值，原因是 NPU 分块累加策略不同。"

**为什么错：**
- O(K x eps) 是**顺序内积**的误差界，前提是 N 个数按固定顺序逐个累加
- CPU BLAS 和 NPU Cube 都是分块 GEMM，累加顺序完全不同，这个界的前提不成立
- 你不知道分块大小、归约拓扑、是否用了 FMA，就无法给出准确的理论上界
- "略高于理论值"实际上比理论值高了 13 倍，这不是"略高"

**正确做法：**
- **不要引用你无法验证前提的理论界。** 除非你确切知道两端的实现细节。
- 改用经验性方法：多平台对比（方法 B）或扰动实验（方法 A）
- 如果要提理论，只说“浮点累加误差随内积长度增长是已知现象”，不要给具体数字

### 坑 2：mean + max 无法推断分布

**原报告写法：**
> "mean_abs 始终保持在 1e-5 量级，说明绝大多数元素误差极小，max_abs 仅体现极端个别值。"

**为什么错：**
考虑两种分布：
- 分布 A：99.99% 元素 diff=0，1 个 outlier diff=0.01 -> mean=1e-6, max=0.01
- 分布 B：10% 元素 diff=1e-5，其余为 0 -> mean=1e-6, max=1e-5

两者的 mean 和 max 可以完全相同，但含义截然不同。A 有局部异常，B 是均匀小偏差。

**正确做法：**
输出分位数（P50, P90, P99, P99.9），它们能区分上述两种分布。见第 7 节脚本规范。

### 坑 3：后设阈值 (post-hoc threshold)

已在第 3 节详述。核心：**阈值的依据必须独立于观测值**。

### 坑 4：把 CPU 结果当“真值”

已在第 0 节详述。实操建议：
- 报告中用“差异 (diff)”而不是“误差 (error)”
- 用“参考平台 (reference)”而不是“基准 (ground truth)”
- 在报告开头明确声明："本报告以 CPU PyTorch 为参考实现，两端差异反映不同硬件的浮点行为差异，而非一端正确一端错误。"

### 坑 5：高 rel_diff 被隐藏

**原报告问题：**
gelu max_rel_diff=0.764 (76%), linear max_rel_diff=0.341 (34%), 但精度总结只写了 "max_abs <= 1.6e-3"。

**为什么这是问题：**
读者看到精度总结只有 max_abs，会认为所有维度都在控制之内。实际上 76% 的相对差异是一个需要解释的信号——即使解释后是可接受的（如“小值放大效应”），也应该在总结中出现。

**正确做法：**
总结表应同时包含 abs_diff 和 rel_diff 的关键统计量。对于高 rel_diff，明确标注其出现条件（如“仅在 |x| < 1e-4 的元素上”），并说明为什么对任务指标无影响（回链 L0）。

### 坑 6：无控制实验的因果归因

已在第 4 节详述。核心：任何"差异来自 X"的声明都需要控制实验支撑。“固定噪声对照”是扩散模型中分离 RNG 影响和计算影响的标准方法。

---

## 6. 报告结构模板

以下模板体现“自顶向下”的验证逻辑。报告的结论应该在第一页就能看到，细节向后展开。

```markdown
# [Model] CPU vs NPU 验证报告

## 测试环境
(同原报告，保留)

## 结论摘要
- 一句话结论：迁移是否正确，依据是什么
- L0 任务指标对比表（benchmark 名称、样本量、指标名称、两端数值、p-value）
- 已知限制和待验证项

## 1. 应用级验证 (L0)
- benchmark 选择依据
- 测试集、样本量、运行次数
- 任务指标对比（含统计检验结果）
- 结论：任务能力是否等价

## 2. 模型级验证 (L1)
- 确定性前向对比（固定输入，比较输出）
- [随机模型] 固定噪声对照实验
- [随机模型] 分布等价性检验
- 误差传播观察（逐层 diff 变化趋势）

## 3. 模块级验证 (L2)
- 关键子模块边界的 diff 统计
- 误差放大/衰减观察

## 4. 算子级验证 (L3)
- 单算子 diff 统计表（含分位数）
- 阈值设定依据（引用方法 A/B/C/D 中的哪一个）

## 5. 性能对比
(同原报告，保留)

## 附录
- 复现命令
- 原始数据 / 日志链接
```

**关键变化说明：**
- L0 放最前，读者第一时间看到"结果对不对"
- L3 放最后，作为支撑材料而非主要论据
- 每个阈值都必须标注依据

---

## 7. 测试脚本输出规范

脚本输出的统计量应该足以支撑报告中的判断。以下是推荐的输出规范。

### 算子级 (L3) 必须输出的统计量

```python
import torch
import numpy as np

def report_diff(name, cpu_tensor, npu_tensor):
    """L3 算子级对比的标准输出"""
    diff = (cpu_tensor.float() - npu_tensor.float()).abs()
    abs_ref = cpu_tensor.float().abs()

    # --- 绝对差异统计 ---
    abs_stats = {
        'max':    diff.max().item(),
        'mean':   diff.mean().item(),
        'P50':    diff.quantile(0.50).item(),
        'P90':    diff.quantile(0.90).item(),
        'P99':    diff.quantile(0.99).item(),
        'P99.9':  diff.quantile(0.999).item(),
    }

    # --- 相对差异统计（排除近零元素） ---
    nonzero_mask = abs_ref > 1e-6  # 避免除零
    if nonzero_mask.any():
        rel_diff = diff[nonzero_mask] / abs_ref[nonzero_mask]
        rel_stats = {
            'max':  rel_diff.max().item(),
            'mean': rel_diff.mean().item(),
            'P99':  rel_diff.quantile(0.99).item(),
        }
    else:
        rel_stats = {'max': 0, 'mean': 0, 'P99': 0}

    # --- 异常元素统计 ---
    outlier_stats = {
        'num_nan':    torch.isnan(npu_tensor).sum().item(),
        'num_inf':    torch.isinf(npu_tensor).sum().item(),
        'pct_gt_1e-3': (diff > 1e-3).float().mean().item() * 100,
        'pct_gt_1e-2': (diff > 1e-2).float().mean().item() * 100,
    }

    # --- 整体相似度 ---
    cosine_sim = torch.nn.functional.cosine_similarity(
        cpu_tensor.float().flatten().unsqueeze(0),
        npu_tensor.float().flatten().unsqueeze(0)
    ).item()

    print(f"=== {name} ===")
    print(f"  Shape: {list(cpu_tensor.shape)}, Dtype: {cpu_tensor.dtype}")
    print(f"  Abs diff: max={abs_stats['max']:.2e}, mean={abs_stats['mean']:.2e}, "
          f"P90={abs_stats['P90']:.2e}, P99={abs_stats['P99']:.2e}, "
          f"P99.9={abs_stats['P99.9']:.2e}")
    print(f"  Rel diff (|ref|>1e-6): max={rel_stats['max']:.2e}, "
          f"mean={rel_stats['mean']:.2e}, P99={rel_stats['P99']:.2e}")
    print(f"  Outliers: NaN={outlier_stats['num_nan']}, "
          f"Inf={outlier_stats['num_inf']}, "
          f">1e-3={outlier_stats['pct_gt_1e-3']:.2f}%, "
          f">1e-2={outlier_stats['pct_gt_1e-2']:.2f}%")
    print(f"  Cosine similarity: {cosine_sim:.8f}")
```

示例输出：

```
=== matmul (8,256,512) fp32 ===
  Shape: [8, 256, 512], Dtype: torch.float32
  Abs diff: max=7.32e-04, mean=7.69e-06, P90=1.43e-05, P99=8.21e-05, P99.9=3.12e-04
  Rel diff (|ref|>1e-6): max=5.43e-01, mean=4.44e-06, P99=2.31e-04
  Outliers: NaN=0, Inf=0, >1e-3=0.00%, >1e-2=0.00%
  Cosine similarity: 0.99999988
```

为什么这比 mean+max 好：
- **分位数** (P90/P99/P99.9) 能区分"少量 outlier" 和 "广泛偏差"
- **排除近零元素的 rel_diff** 避免"小值放大效应"干扰
- **异常元素统计** 能立刻发现 NaN/Inf 类的硬件 bug
- **cosine similarity** 提供一个单一数字的整体相似度判断

### 模型级 (L1) 推荐输出

```python
def report_model_diff(name, cpu_output, npu_output):
    """模型输出层的对比"""
    # 基础数值对比（同 report_diff）
    report_diff(name, cpu_output, npu_output)

    # 额外：检查功能等价性
    # 例如分类模型：top-k 一致率
    cpu_topk = cpu_output.topk(5, dim=-1).indices
    npu_topk = npu_output.topk(5, dim=-1).indices
    top1_agree = (cpu_topk[:, 0] == npu_topk[:, 0]).float().mean()
    top5_agree = sum((cpu_topk[:, i] == npu_topk[:, i]).float().mean()
                     for i in range(5)) / 5
    print(f"  Top-1 agreement: {top1_agree:.4f}")
    print(f"  Top-5 agreement: {top5_agree:.4f}")
```

### 分布等价性检验（随机模型用）

```python
from scipy import stats

def report_distribution_equivalence(cpu_scores, npu_scores, metric_name):
    """比较两个平台多次运行的指标分布"""
    cpu_arr = np.array(cpu_scores)
    npu_arr = np.array(npu_scores)

    print(f"=== {metric_name} 分布对比 ===")
    print(f"  CPU: mean={cpu_arr.mean():.4f}, std={cpu_arr.std():.4f}, "
          f"median={np.median(cpu_arr):.4f}, N={len(cpu_arr)}")
    print(f"  NPU: mean={npu_arr.mean():.4f}, std={npu_arr.std():.4f}, "
          f"median={np.median(npu_arr):.4f}, N={len(npu_arr)}")

    # Mann-Whitney U 检验（不假设正态分布）
    stat, pval = stats.mannwhitneyu(cpu_arr, npu_arr, alternative='two-sided')
    print(f"  Mann-Whitney U test: stat={stat:.1f}, p={pval:.4f}")
    print(f"  Conclusion: {'NO significant diff (p>0.05)' if pval > 0.05 else 'SIGNIFICANT diff (p<=0.05)'}")
```

---

## 8. 报告完成后的自查清单

写完报告后，逐项检查。**任何一项为"否"都应补充后再发布。**

### 方法论层

- [ ] 报告是否包含 L0 应用级验证？（有 benchmark 指标对比，而不只是"能跑通"）
- [ ] 结论是否基于 L0 而不是 L3？（“迁移正确”的依据是任务指标，不是算子 allclose）
- [ ] L0 样本量是否足够做统计推断？（单个样本不够）
- [ ] [随机模型] 是否做了固定噪声对照实验？
- [ ] [随机模型] 是否做了分布等价性检验（多次运行 + 统计检验）？

### 阈值和统计量

- [ ] 每个阈值是否有独立于观测值的依据？（标注了方法 A/B/C/D 中的哪一个）
- [ ] diff 统计是否包含分位数（P90/P99/P99.9），而不只有 mean+max？
- [ ] 高 rel_diff 是否在总结中显式提及并解释？
- [ ] 是否检查了 NaN/Inf 异常值？

### 语言和表述

- [ ] 是否用"差异 (diff)"而不是"误差 (error)"？
- [ ] 是否用"参考平台 (reference)"而不是"基准/真值 (ground truth)"？
- [ ] 是否避免了引用无法验证前提的理论界？
- [ ] 因果声明（"差异来自 X"）是否有控制实验支撑？

### 可复现性

- [ ] 是否提供了完整的复现命令？
- [ ] 是否固定了所有随机种子？
- [ ] 环境信息是否完整（包含 CANN 版本、驱动版本）？

---

## 9. 回到核心问题："参照什么"

回答报告作者的困惑：

> "目前来看 npu 上他适配的版本按照 readme 的推理流程是能跑通，结果也正常出。我纠结的点在于跑出来的结果是不是对，不知道参照什么。"

答案是分层的：

1. **你的第一参照是模型的标准 benchmark。** Boltz2 的参照是 CAMEO/CASP 上的 TM-score。在上面跑一遍，看 NPU 的分数和 CPU 的分数是否等价。如果等价，就是对的。

2. **你的第二参照是已验证平台的 diff 量级。** 如果你能同时拿到 CUDA GPU 的结果，GPU-CPU diff 就是你的参照线。NPU-CPU diff 在同一量级就是合理的。

3. **你的第三参照是模型自身的数值敏感度。** 通过扰动实验确定模型能容忍多大的数值差异，只要你的 diff 在容忍范围内，就是对的。

4. **兆底参照是业界惯例。** 如果以上都做不到，至少引用 ONNX Runtime/TensorRT 的公开容忍度作为 baseline。

“不知道参照什么”的真正原因是：把 CPU 输出当作了唯一参照，然后发现两端不一样，就不知道该信谁了。

解法是：**跳出 CPU vs NPU 的二元对比，把视角拉高到任务层。** 在任务层有客观的参照（benchmark），在数值层有多个交叉参照（多平台、扰动实验、业界惯例）。

不需要一个"标准答案"，需要的是一套多维度的证据体系。

