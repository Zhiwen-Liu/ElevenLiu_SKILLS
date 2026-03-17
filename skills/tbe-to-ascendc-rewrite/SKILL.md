---
name: tbe-to-ascendc-rewrite
description: Rewrite Huawei TBE operators as AscendC native kernels for Ascend NPU. Use when users want to convert TBE operators to AscendC, develop/debug/optimize AscendC kernels, fix precision/performance issues in Ascend operator development, or design multi-template kernel architectures for NPU operators.
---

# TBE 算子 AscendC 改写

这个 skill 指导将 TBE 算子迁移为 AscendC 原生 kernel。核心理念是「先正确后快」：先实现一个功能完备的泛化模板，验证精度后，再通过多模板架构针对不同 shape 场景做性能优化。

---

## 改写流程

```
分析TBE原始实现 → 查官方参考 → 泛化模板实现 → 精度验证 → 多模板优化 → 性能验证 → 交付
```

每次优化后必须重跑精度，因为同步原语和流水线的微小调整很容易引入数据竞争。

### 官方参考查找

官方宗宗算子库 https://gitcode.com/cann/ops-nn 是最重要的参考源。查找同类算子时关注：

- `op_host/` 中的 `CalTilingKey()` — 理解它如何根据shape选择模板
- `op_kernel/` 中同一算子的多个 `.h` 文件 — 每个对应一个模板分支
- 命名规律：`_float.h` / `_cast.h`（按数据类型分），`_nc_large_*.h`（按UB容量分）

---

## 工程规范

```
OpName/
  op_host/     # Tiling结构体(.h) + tiling计算(.cpp)
  op_kernel/   # AscendC kernel
  torch_ext/   # torch C++ extension，用于性能对比
  test/        # 测试脚本
  results/     # 测试结果
```

- Tiling 结构体只在 `op_host/` 定义一次，kernel 通过 `#include` 引用
- Tiling 结构体只保留 kernel 实际引用的字段，避免迭代过程中残留废弃字段增加传输开销和维护负担

---

## 多模板架构设计

NPU算子的标准开发模式是：**先做一个功能泛化的模板支持所有场景，再根据实际shape增加性能优化模板，在tiling里判断走哪个模板**。这个模式不仅适用于池化算子，几乎所有涉及NC维度和空间维度的算子都遵循同样的分支逻辑。

### 模板分支的常见维度

多模板的分支维度通常有两个，它们可以组合：

**维度一：UB容量（几乎所有算子都需要）**

判断处理一个位置所需的数据能否装进UB：
```
perCalcSize = 所需buffer数 × ncAlign × elemSize
if perCalcSize ≤ ubSize → SMALL模板（整体装入，可批量处理）
else                    → LARGE模板（切片处理）
```

SMALL模板可以通过 `yNumPerCalc = ubSize / perCalcSize` 批量处理多个空间位置，减少同步开销。
LARGE模板需要将NC维度切片，task数膨胀为 `空间位置数 × NC切片数`，每个task处理一个（空间位置, NC片）。

**维度二：数据类型（涉及fp16/bf16时需要）**

fp32可以直接计算，fp16/bf16通常需要cast到fp32再计算（尤其是涉及累加/原子操作时），这意味着UB需要额外的cast buffer，perCalcSize也随之变化。

官方用 `tilingKey = dtype编码 * 10 + UB容量编码` 组合两个维度，kernel通过 `TILING_KEY_IS(n)` 分发到对应模板。

### NC切片的通用策略

LARGE模板的NC切片逻辑是通用的：
```
ncSliceNum = ceil(perCalcSize / ubSize)                            // 初始切片数
ncSliceLen = ncAlign / ncSliceNum / ALIGN_NUM * ALIGN_NUM           // 每片对齐后大小
ncSliceNum = ceil(ncNum / ncSliceLen)                               // 重算实际片数
ncSliceTail = ncNum - ncSliceLen * (ncSliceNum - 1)                 // 尾片实际长度
taskNum = spatialPositions * ncSliceNum                             // task数膨胀
```

kernel中task拆解：
```
n = taskIdx % ncSliceNum            // NC第几片
spatialIdx = taskIdx / ncSliceNum   // 空间位置
ncMoveNum = (n == ncSliceNum-1) ? ncSliceTail : ncSliceLen
```

---

## 同步原语选择

AscendC提供两级同步机制，选择不当会严重影响性能或导致精度错误。

### PipeBarrier（粗粒度，安全但慢）

`PipeBarrier<PIPE_ALL>` 阻塞所有pipe，最安全但完全消除了流水线并行性。适合调试阶段用来确认逻辑正确，不适合交付代码。

### HardEvent（细粒度，快但需精确配对）

HardEvent用 `SetFlag/WaitFlag` 控制特定pipe间的依赖，允许无关pipe并行执行，通常带来20-50%性能提升。

NPU的三级流水线是 MTE2(读) → V(计算) → MTE3(写)，对应四种event：

| Event | 含义 | 典型用法 |
|-------|------|----------|
| MTE2_V | 读完成，V可计算 | DataCopy后 Set，Muls前Wait |
| V_MTE3 | 计算完成，可写出 | Muls后Set，scatter前Wait |
| V_MTE2 | V释放input buffer | Muls后Set，下轮DataCopy前Wait |
| MTE3_V | 写出完成，可复用output buffer | scatter后Set，下轮Muls前Wait |

关键规则：
- Init中Alloc，Release中Release，必须成对
- 外层循环开始前要先SetFlag一次作为初始信号
- Process结束前要WaitFlag所有未完成的event
- 如果精度出问题，先全换成PipeBarrier<PIPE_ALL>定位问题，确认逻辑正确后再逐步替换回HardEvent

### 流水线模式

SMALL模板的批次间流水——上一批的MTE3与下一批的MTE2可以重叠：
```
外层循环(按yNumPerCalc分批):
  WaitFlag<V_MTE2>; WaitFlag<MTE3_V>;     // 等上一批完成
  内层处理每个task...
  SetFlag<V_MTE2>; SetFlag<MTE3_V>;       // 通知本批完成
```

LARGE模板每个task间流水——MTE2与上一task的MTE3重叠，效率更高：
```
每个task:
  WaitFlag<V_MTE2>;       DataCopy(in←GM);   SetFlag<MTE2_V>;
  WaitFlag<MTE2_V,MTE3_V>; Muls(out,in,fac);  SetFlag<V_MTE2,V_MTE3>;
  WaitFlag<V_MTE3>;       scatter(GM←out);   SetFlag<MTE3_V>;
```

---

## 算子常见优化手段

按优先级排序，前三项几乎每个算子都应考虑：

1. **HardEvent替代PipeBarrier** — 通常是最大的单项性能提升
2. **分支跳过不必要的开销** — 可整除时跳过atomic和clear，输入输出同尺寸时跳过整个kernel等
3. **批量处理** — 小数据量kernel的launch开销不可忽视，yNumPerCalc把多次计算合并到一次buffer分配中
4. **NC切片而非gather** — 大NC场景保持scatter模式通常比gather更稳定，因为scatter的GM写入模式更简单
5. **合并kernel调用** — ClearOutput+Process在同一kernel函数中顺序执行，减少launch开销
6. **Tiling结构体精简** — 删除不用的字段减少host→device传输量

---

## 内存与对齐

### DataCopy对齐

所有DataCopy操作的传输量必须对齐到32字节（fp32对齐8元素，fp16对齐16元素）。NC不对齐时有两种处理方式：

- **host侧padding**：将NC pad到ncAlign，kernel按ncAlign操作，输出后裁剪。通用但增加host开销。
- **DataCopyPad**：高版本SDK提供的API，直接处理非对齐数据，避免host padding。如果SDK不支持则退化到第一种。

### UB容量保护

UB总容量约192-256KB（平台相关），需要预留系统开销（通常减去6KB）。当NC维度随输入变化时，必须在tiling中计算实际分配量，不能写死UB大小。

---

## 散射与原子操作

当多个output window重叠覆盖同一个input位置时，多核scatter会产生写入冲突，需要原子操作。关键判断和处理路径：

- **是否可整除**：当输入尺寸能被输出尺寸整除时，每个input位置恰好被一个output覆盖，无冲突，跳过atomic和clear
- **不可整除**：必须SetAtomicAdd，且必须先ClearOutput再散射，两个阶段之间需要SyncAll全核同步
- **fp16原子累加精度不足**：官方做法是用fp32 workspace累加，最后cast回。简化做法是在host侧转成fp32计算

---

## 精度验证规范

以TBE算子为对比基准，容差 fp32 atol=1e-5, fp16 atol=1e-2。

测试用例必须覆盖的维度：

| 维度 | 目的 | 示例 |
|------|------|------|
| 模板分支 | 确保每个tiling分支都被测到 | 小NC + 大NC |
| atomic路径 | 整除和非整除都要覆盖 | 56/7=整除, 7/3=非整除 |
| 规模范围 | 小/中/大 | scalar, 典型模型, 极端大NC |
| 边界值 | 对齐边界、特殊尺寸 | NC=1, identity(in=out), 素数HW |
| 数据类型 | 每个场景双类型 | fp32 + fp16 |

---

## 性能验证规范

- 随机生成1000个用例，要求≥95%不劣于TBE
- **两边必须对等**：都通过torch dispatch调用，禁止ctypes直调与torch dispatch对比
- **预分配缓冲区**：计时循环内不应包含tensor分配
- **报告包含分布统计**：加速比的最差/P5/中位/P95/最佳，以及每个模板的胜率

---

## 踩坑速查

| 现象 | 排查方向 |
|------|----------|
| 崩溃/挂死 | UB溢出 → 查极端 shape 下的分配; DataCopy越界 → 查对齐后是否超tensor边界 |
| 部分输出全零 | 多核协作错误 → 查任务分配; ClearOutput后SyncAll缺失 |
| 精度偏差 | 分支不对齐 → 逐行对照TBE; fp16累加 → 用fp32计算; in-place Muls → 改非in-place或加barrier |
| HardEvent后精度错 | SetFlag/WaitFlag配对遗漏 → 先全换PipeBarrier<PIPE_ALL>定位，确认正确后再逐步替换回 |
| 性能数据失真 | 调用层级不对等 → 确认都走torch dispatch; 循环内分配 → 预分配 |
| 编译后行为未变 | 编译缓存 → clean重建; torch ext未重编 → 重跑setup.py |

---

## 交付检查清单

- [ ] Tiling结构体无废弃字段，与kernel引用完全一致
- [ ] 精度测试覆盖所有模板分支、atomic/非atomic路径、边界值、双数据类型
- [ ] 性能 1000 随机用例 ≥95%，报告含分模板胜率和加速比分布
- [ ] 性能测试两边均通过 torch dispatch
- [ ] README与代码一致，无旧数据、无残留中间产物
