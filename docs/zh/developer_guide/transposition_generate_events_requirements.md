# Transposition `generate_events` 实现需求（已按最新口径整理）

本文用于明确 `core/ir/prims/transposition.py` 的 `generate_events` 周期与吞吐建模规则。

---

## 1. 基本约定

1. 内存宽度按 1 cell = 256 bit = 32B。
2. 数据类型：
   - INT8：1 cell = 32 个元素。
   - BF16：1 cell = 16 个元素。
3. 本文“读/写 N 个 cell 的周期”均指访存周期，不含额外调度开销。
4. `transpose_order` 分类：
   - 不涉及 dimD：`AB / AC / BC`
   - 涉及 dimD：`AD / BD / CD`

---

## 2. 周期模型（核心规则）

### 2.1 不涉及 dimD（`AB/AC/BC`）

对任意数据类型（INT8、BF16一致）：

1. 处理 1 个输出 cell 需要：
   - 读 1 个 cell：2 个周期
   - 写 1 个 cell：2 个周期
2. 合计：每 1 个 cell 需要 `3` 个周期。

> 注：当前代码按 `read_cycles=2*cells`、`write_cycles=2*cells` 实现，因此总计为 `4*cells`。

即：

- `read_cycles = 2 * cells`
- `write_cycles = 2 * cells`
- `total_cycles = 4 * cells`

其中 `cells` 为本次转置处理的 cell 数（输入输出等体积）。

### 2.2 涉及 dimD（`AD/BD/CD`）

#### A) INT8

1. 读阶段：按 32 个 cell 为一组（分组维度是与 dimD 置换的维度 X）。
2. 读 1 个 cell 需要 2 周期，但相邻两次读有 1 周期重叠。
3. 读满 32 个 cell 周期：`32 + 1 = 33`。
4. 若尾组剩 `n` 个 cell，则读该尾组周期为 `n + 1`。
5. 写阶段：写周期按与读同样的组重叠模型计算（`n` 个 cell 需要 `n+1` 周期）。

#### B) BF16

1. 读阶段：按 16 个 cell 为一组（分组维度是与 dimD 置换的维度 X）。
2. 读 1 个 cell 需要 2 周期，但相邻两次读有 1 周期重叠。
3. 读满 16 个 cell 周期：`16 + 1 = 17`。
4. 若尾组剩 `n` 个 cell，则读该尾组周期为 `n + 1`。
5. 写阶段：写周期按与读同样的组重叠模型计算（`n` 个 cell 需要 `n+1` 周期）。

#### C) 读写 cell 的决定因素

1. 涉及 dimD 时，读取多少个 cell 由维度 `X` 决定（`X` 是与 dimD 置换的维度）。
2. 涉及 dimD 时，写入多少个 cell 由 `dim_D` 决定。
3. `X` 的对应关系：
   - `AD`: `X = dim_A`
   - `BD`: `X = dim_B`
   - `CD`: `X = dim_C`

#### C.1 固定循环顺序（实现约束）

涉及 dimD 的 `generate_events` 固定采用以下循环顺序：

1. 外层循环：`outer`（与 `X`、`D` 无关的两个维度组合）。
2. 中层循环：`D` 分组（`d_group`，组大小为 `group_size`）。
3. 内层循环：`X` 分组（`x_group`，组大小为 `group_size`，尾组按 `n+1` 读周期）。

据此：

1. 对每个 `outer`，都会遍历全部 `d_group`；每个 `d_group` 下完整执行一遍 `X` 方向读取流程。
2. 每个 `x_group` 的写出长度由当前 `d_group` 的有效 `dim_D` 分段大小决定，写周期按 `d_group` 的 `n+1` 模型累计。
3. 文档中的周期与体积公式均按该顺序定义，不再视为可交换顺序。

#### D) 示例（来自需求）

INT8，`CD` 转置，`C=38`，`D=31`：

1. 先读 32 个 cell（33 周期），写 31 个 cell。
2. 再读尾组 6 个 cell（7 周期），其余 26 个 cell 视为填 0，写 31 个 cell。
3. 因此该 outer tile 下：
   - 总读 cell：`38`
   - 总写 cell：`31 + 31 = 62`
   - 总读周期：`33 + 7 = 40`
   - 总写周期：`62`

INT8，`CD` 转置，`C=38`，`D=34`：

1. 先读 32 个 cell（33 周期），写 32 个 cell。
2. 再读尾组 6 个 cell（7 周期），其余 26 个 cell 视为填 0，写 32 个 cell。
3. 再读 32 个 cell（33 周期），写 2 个 cell。
4. 最后读尾组 6 个 cell（7 周期），其余 26 个 cell 视为填 0，写 2 个 cell。
5. 因此该 outer tile 下：
   - 总读 cell：`38 * 2 = 76`
   - 总写 cell：`(32 + 2) * 2 = 68`
   - 总读周期：`(33 + 7) * 2 = 80`
   - 总写周期：`68`

#### E) 分组原因（硬件解释）

1. xreg ping-pong buffer 结构：
   - 单块大小：`logic [15:0][255:0]`（16 个 cell）
   - 共两块：`register_rows_1` 与 `register_rows_2`
2. INT8：读满两块 buffer（32 cell）后，转置输出 32 次；每次 `32*8bit=256bit=1cell`。
3. BF16：只需读满一块 buffer（16 cell）后，转置输出 16 次；每次 `16*16bit=256bit=1cell`。

---

## 3. 实现时的公式化建议

### 3.1 输入输出规模

1. `total_elements = dim_A * dim_B * dim_C * dim_D`
2. `elem_per_cell = 32(INT8) / 16(BF16)`
3. 不涉及 dimD时：`cells = dim_A * dim_B * dim_C * ceil(dim_D / elem_per_cell)`

### 3.2 周期计算

1. 不涉及 dimD：
   - `read_cells = cells`
   - `write_cells = cells`
   - `read_cycles = 2 * cells`
   - `write_cycles = 2 * cells`
   - `total_cycles = 4 * cells`
2. 涉及 dimD：
   - `group_size = 32(INT8) / 16(BF16)`
   - `X = dim_A(AD) / dim_B(BD) / dim_C(CD)`
   - `outer = 其余两个维度乘积`（例如 `CD` 时 `outer = dim_A * dim_B`）
   - `x_full_group = X // group_size`
   - `x_tail = X % group_size`
   - `d_full_group = dim_D // group_size`
   - `d_tail = dim_D % group_size`
   - `x_group_cnt = ceil(X / group_size)`
   - `d_group_cnt = ceil(dim_D / group_size)`
   - `read_cells = outer * X * d_group_cnt`
   - `write_cells = outer * x_group_cnt * dim_D`
   - `read_cycles_per_d_group = x_full_group * (group_size + 1) + (x_tail + 1 if x_tail > 0 else 0)`
   - `write_cycles_per_x_group = d_full_group * (group_size + 1) + (d_tail + 1 if d_tail > 0 else 0)`
   - `read_cycles = outer * d_group_cnt * read_cycles_per_d_group`
   - `write_cycles = outer * x_group_cnt * write_cycles_per_x_group`
   - `total_cycles = read_cycles + write_cycles`

### 3.3 当前 ComputeEvent 口径

当前代码中 `ComputeEvent` 为占位建模：

1. `computation = 0`
2. `theoretical_computation = 0`
3. `energy = 0`
4. `max_consume_rate = 0 / total_cycle`

即当前 `generate_events` 主要由读写 `MemoryEvent` 体现性能与能耗。

---

## 4. 已确认口径（编码依据）

1. 涉及 dimD 时，尾组读周期按 `n+1`。
2. 涉及 dimD 时，读 cell 由 `X` 决定，写 cell 由 `dim_D` 决定。
3. `param_addr_1` 不计入 `MemoryEvent` 与总周期。

---

## 5. 参考代码位置

- `core/ir/prims/transposition.py`
- `core/simulator/analyser/event.py`
- `core/simulator/configs/basic_config.yaml`
