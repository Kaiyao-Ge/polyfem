# 外部计数工具：导出运行 → 重放定价 → 估算

PolyFEM 正式代码里没有日志、计时类别和导出。工具在 PolyFEM 外面：读 `sim.json` 的计数器，乘本机上测的单价，估算各项成本：

    估算 = Σ 项（计数 × 单价）

`sim.json` 的计数器够做"次数类"的估算；"规模类"（每次分解的 flops 与 lnz、每个接受点的块数、精确单元数和精确对数、CCD 候选数、交给 Tight Inclusion 的查询）要另编的**导出构建**（`tools/nonsmooth_counting/export/`）跑一遍才有。没有导出时 `estimate.py` 退回平均单价，并在输出里注明每一行用的是哪种依据。

目录：

| 路径 | 内容 |
|---|---|
| `estimate.py` | 估算脚本，只用 Python 标准库（Linux、Mac 都能直接跑） |
| `replay/` | 重放定价程序和拟合脚本（`cl/B` 修过的版本，见下）。`build_replays.sh` 编译，`run_replay.sh` 跑，`fit_replay.py` 出单价文件 |
| `../export/` | 导出树和导出构建（`README.md`） |

## 1 导出运行（Linux）

1. 导出构建（一次）：

       tools/nonsmooth_counting/export/build.sh        # 重任务；有内存上限的机器放在上限里跑，ninja -j2

2. 跑场景，两个开关都设（导出构建的二进制 `$EXPORT_WORK/build/PolyFEM_bin`）：

       RUN=/some/dir/mat3k_45
       mkdir -p $RUN
       POLYFEM_CL_COUNTS=$RUN POLYFEM_CL_EXPORT=$RUN/export MKL_CBWR=COMPATIBLE \
         $EXPORT_WORK/build/PolyFEM_bin -j <场景>.json -o $RUN --max_threads 1

   产物：`$RUN/sim.json`（计数器和 PolyFEM 的墙钟计时）、`$RUN/cl_counts.jsonl`（规模数据）、`$RUN/export/`（重放用的导出）。开关不设，二进制的解和计数器与正式二进制逐位相同；设了，解也相同。

   盲选用 JSON 参数 `solver/nonlinear/NonsmoothNewton/selection = "blind"`（见 `docs/nonsmooth-newton.md`），其他不变。能量分项（c4R2）是默认。要导出 `elem`（精确弹性单元，元素核的单价靠它）需要 `solver/advanced/fast_p1_kernel = true`（闭式 P1 核，平台选项，默认关）：没有它 `elem` 不导出，G1、E1、V1 没有元素核单价，不估算。能量分项没有精确单元，所以元素核的单价要从盲选运行（或开了第一层的运行）的导出里来；`run_replay.sh` 可以同时给几个运行的导出目录，价按所有记录拟合。

## 2 重放定价（带金丝雀）

单价要在**要估算的那台机器**上测。`run_replay.sh` 依次做：金丝雀（测机器当时的速度）→ 重放程序 → 金丝雀 → 拟合，输出 `unit_costs_cl.json`。

    tools/nonsmooth_counting/counting/replay/build_replays.sh                       # 编译三个程序（一次）
    tools/nonsmooth_counting/counting/replay/run_replay.sh --out $PRICES $RUN/export [其他运行的 export 目录 ...]

- `$PRICES/unit_costs_cl.json`：单价文件（`fit_replay.py` 的输出格式，`estimate.py` 读它）。可以给多个运行的导出目录，价按所有记录拟合。
- 重放程序读导出文件，不依赖 PolyFEM 的日志：
  - `cl_replay_solver l4a`：CHOLMOD 分解（成功的和失败的）、符号分析、回代；`l4c`：约化矩阵 A 的合并写法和逐块写法在真实块布局上的时间；`l4e`：小运算（特征分解、排序、spmv、gemv、k×k 解、取行）；
  - `cl_replay_ipc l4b`：宽相（每次新建宽相对象）；窄相：标准路径在全部缓存候选上跑 TI，证书路径只重放导出的"真正交给 Tight Inclusion 的查询"（`tiq` 记录）加一遍证书；碰撞集合构建（自带宽相的，和从缓存候选构建的）；`l4d`：弹性核与接触核。
- 金丝雀在运行之间跑两次，记录机器的计算、带宽、访存速度。两次读数相差超过 10%，说明测的过程中机器速度变了，这一轮价不可信，重测。单价的 `meta.canary_before/after` 记着读数；不同机器的价不要混用。
- 这是重任务，每个程序单线程。本机（Linux，有共用锁和内存上限的规则）：

      CL_WRAP="<内存上限的命令前缀，可不设>" CL_PIN="taskset -c 3" tools/nonsmooth_counting/counting/replay/run_replay.sh --out $PRICES $RUN/export

  没有这些规则的机器（Mac）：什么也不设，跑之前关掉其他大任务。`CL_RUN_L4B=0` 跳过 CCD 重放（C1–C3 就不定价）。

## 3 估算

    python3 tools/nonsmooth_counting/counting/estimate.py --sim $RUN/sim.json --prices $PRICES/unit_costs_cl.json --counts $RUN/cl_counts.jsonl

- `--counts` 可省：省了就是"没有导出"的估算（平均单价；C1–C3、E1、E2、A1 没有规模数据，不估算，输出里写明；C1–C3 旁边列 PolyFEM 自己的墙钟计时）。
- `--n-elements N`：场景的弹性单元数（默认取 `sim.json` 的 `num_elements`）。
- `--l4c-mode hot|cold|mid`：约化矩阵核的单价取热缓存、冷缓存，或二者的几何平均（默认 `mid`）。
- `--json FILE`：同时写 JSON。

输出每一项（F1 数值分解、F2 符号分析、F3 回代、C1 宽相、C2 窄相、C3 碰撞集合、G1 光滑部分组装、G2 重读、E1 精确弹性、E2 精确接触、A1 约化矩阵、A2 子空间代数、S1 选择、V1 真实能量、H1 Newton 链组装）的估算秒数、依据（有导出用规模量，没有用平均单价或不估算）、有对应的 PolyFEM 墙钟计时就并排列出；最后是各项之和与 `sim.json` 里 `total_time` 的比值（只作参考：估算缺没有数字的项和项目之外的一切，测量是整个运行的墙钟）。

`example/` 里有 mat3k_45（能量分项 c4R2 和盲选）、bowl_19_it1000 的单价文件和估算输出，作格式参考（不同机器、不同时刻的价不要拿来用）。

估算给出的对照列：`PolyFEM timer`（`sim.json` 里 PolyFEM 自己的计时 × 迭代数；方法路径上的宽相、窄相、碰撞集合构建不在这些计时里，所以它不是 C1–C3 的检验）和 `export run cpu`（导出运行自己在这些操作外面测的线程 CPU 时间：F1、F2、C1、C2、C3）。后者是检验单价的：估算 / 导出运行的实测，mat3k_45 上 F1 0.97–0.98、F2 0.96、C1 1.05、C2 0.80–0.85、C3 1.02；bowl_19_it1000 上 F1 1.07、F2 0.99、C1 1.12、C2 0.88、C3 0.93。

项与计数的对应见 `counting_tool.md` 第 2 节。估算脚本里的近似（都写在输出的"依据"一列）：

- G1：每个"有 B 的外层"组装一次光滑部分，单位数 = 弹性单元数（盲选时有导出才扣去精确单元），按重放的元素核（`eval_grad_hess`，投影时加 `project_numeric`）定价；
- G2：每次重读 = 一次 spmv（分解矩阵的 nnz）加 `Qᵀ r`（n×k）；
- A2：每个内层步 = 3 次 gemv + 3 次 gemvT（n×k）加一次 k×k 的稠密解；
- V1：每次真实能量求值 = 全部弹性单元的值核加碰撞对的值核（后者要导出）；
- H1：每次链迭代 = 一次投影 Hessian 的组装（单元的 `project_numeric` 加接触对的梯度与投影 Hessian）。
- S1（选择）只列计数，不定价。

## 4 在 Mac 上

- `estimate.py` 不需要 PolyFEM，不需要 numpy，直接跑。
- 导出运行要在装了 PolyFEM 构建依赖的机器上：用 `export/sync.sh`（`PF_SRC`、`PS_SRC` 指向 Mac 上 `blind-nonsmooth` 分支的检出）重新生成钩子树，再按 `export/README.md` 构建。**Mac 的计数和单价只和 Mac 的参考比**；MKL（Linux/x86）与 Accelerate、编译器的融合规则不同，末位不同。bowl 一类混沌场景要用多个种子的分布，不比单次运行。
- 单价要在 Mac 上自己测：重放程序要重新编译。`build_replays.sh` 用 `ninja -t commands` 取 PolyFEM 构建的编译参数和链接库；Mac 上链接行里有 Linux 专有的东西（`-lrt`、`libpthread.a`、MKL 静态库），要改成 Accelerate（`-framework Accelerate`；`EIGEN_USE_MKL_ALL` 关掉，`cl_kernels.hpp` 里同理）和 Homebrew 的 SuiteSparse（`brew install suite-sparse`）。线程 CPU 时间用 `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`（macOS 10.12 起有），代码里已经这样用；金丝雀对 Linux 专有的 `/proc`、`/sys` 读数在 Mac 上给空值（`sibling_busy = -1`），不影响单价。
- 脚本里的 Linux 东西：`taskset`（Mac 没有：不绑核，跑之前关掉其他大任务）、`flock` 和 `capped.sh`（本机的规则，Mac 上不要设 `CL_WRAP`）、GNU `time -v`（`brew install gnu-time` 后用 `gtime`，本工具的脚本不依赖它）。
- 防睡眠：长任务前 `caffeinate -i`。
- `fit_replay.py` 用 numpy 和 scipy（`pip3 install numpy scipy`），只在拟合单价时要。

## 5 `cl/B` 原件和这里的差别

`replay/` 是 `joint/cl/B/` 的副本，`cl/B` 原件没动。改了：

- **宽相重放超过内存上限被杀的原因**：不是"宽相对象反复使用，内存累积"。`l4b` 里 `V1` 和 Tight Inclusion 对象放在 `if (sf.kind == 0) { ... }` 块里，各个计时臂（按引用捕获的 lambda）在块结束之后才运行：use-after-scope。宽相和窄相读的是已经释放的内存，坐标是垃圾；第二轮起候选数就不是第一轮的了（同一条记录，第一次 63 964 个候选，臂里 64 764 个），宽相在垃圾几何上建出巨大的候选集合，内存耗尽（7G 上限下 3 秒内被杀）；Tight Inclusion 在垃圾坐标上报 "overflow occured when splitting intervals"，日志写出 5 GB。"每次新建宽相对象"没有用，因为问题不在宽相对象。现在 `V1` 和 `tight` 在块外，和臂同寿命。修好后 bowl 记录上宽相 5.5 ms、窄相（全部候选）9 ms，内存 36 MB，没有溢出日志。
- 窄相只重放导出里真正交给 Tight Inclusion 的查询（`tiq`），再加证书那一遍（`narrow_cert_pass`）；不再在全部候选上直接跑 TI。标准路径（`narrow=stock`，Newton 链的线搜索）仍在全部缓存候选上跑，那正是求解器做的事。
- 加了从缓存候选构建碰撞集合的臂（`collisions_cached`）。
- `FastP1.hpp` 取自构建的 include 路径（正式的 `FastP1.hpp` 只有 `eval`，没有实验树的 `project_f`、`eval_f`，这两个臂用 `-DCL_FASTP1_FULL` 才编）。
- 路径、锁、内存上限不再写死（`CL_BUILD_DIR`、`CL_WRAP`、`CL_PIN`）。

## 6 全覆盖估算（类别模式）

第 3 节的估算按"项 × 计数"做，覆盖不到整个求解。类别模式把导出运行里**每一段**求解的时间都归到一个类别（`export/README.md` 的"类别计时"），给每个类别定价，加起来对照这次运行自己的线程 CPU 时间（扣掉导出钩子的 `X.hook`）。**测量只用来核对；单价全部来自孤立的重放或微基准。**

    # 1. 导出运行（两个开关都设）：重放要的文件（矩阵、块、元素、CCD 调用、全部 TI 查询）和规模数据
    POLYFEM_CL_COUNTS=$RUN POLYFEM_CL_EXPORT=$RUN/export build/PolyFEM_bin -j scene.json -o $RUN --max_threads 1
    # 2. 被核对的运行：同一场景，只开计数和类别计时、不写导出文件（export/run_with_hooks.sh 加 CL_NO_EXPORT=1）。轨迹与 1 相同；
    #    导出文件的写出会冲掉缓存，让求解变慢，bowl 上钩子占总 CPU 的四分之一，不写文件时占得少得多，测量更接近干净运行
    CL_NO_EXPORT=1 export/run_with_hooks.sh -j scene.json -o $CRUN --max_threads 1
    # 3. 同一场景的 bench 运行，单独一次（POLYFEM_CL_BENCH=<有接触的时间步>）：问题侧每个操作孤立重复，写 bench 行
    CL_NO_EXPORT=1 POLYFEM_CL_BENCH=18 export/run_with_hooks.sh -j scene.json -o $BRUN --max_threads 1
    # 4. 重放定价（第 2 节；含 tiq_all 的全部 TI 查询、冷缓存回代、路径内稠密代数的操作序列）
    replay/run_replay.sh --out $PRICES $RUN/export
    # 5. 估算（--export 指向第 1 步的导出目录：全部 TI 查询的重放按导出目录取）
    python3 estimate.py --sim $CRUN/sim.json --prices $PRICES/unit_costs_cl.json --counts $CRUN/cl/cl_counts.jsonl --export $RUN/export --bench $BRUN/cl/cl_counts.jsonl

bench 的时间步要选有接触的步（接触对数不为 0）：接触相关类别的价按碰撞对数计，bench 状态里没有碰撞对时价是 0。

价从哪来：

| 类别 | 价 |
|---|---|
| `F1`、`F2`、`F3`（含链、选择里的回代） | 重放：CHOLMOD 在导出的矩阵上，分解按 flops，失败的按已做的 flops，符号分析按 nnz，回代按 lnz（热缓存和冷缓存的几何平均） |
| `A1` | 重放：导出的块，当前写法或合并写法 |
| `C1.build`（宽相本身）、`C2.narrow`（窄相本身）、`C3.build`（碰撞集合构建本身，不管谁调用）；它们外面的 `form.barrier-contact.ls_begin`、`NL.max_step_size`、`NT.update`、`form.barrier-contact.solution_changed` 是胶水，走 bench | 重放：宽相按顶点和候选，窄相按证书遍历（候选数）加**这次运行全部 TI 查询的重放**（`tiq_all.bin`），碰撞集合按候选 |
| 问题侧其余（光滑部分的组装、各形式的能量与梯度、有效性、精确项的求值、映射……） | bench 行：孤立重复的独占时间，热缓存和冷缓存的几何平均，按单位数（单元数、碰撞对数、顶点数、非零元数）或调用数 |
| 选择 `S1` | bench 运行里对每次选择在同一数据上的重复（`S1.rep`），按选择表的大小 |
| 路径内的稠密代数（`P.run`、`A2.*`、`G2.set`） | 重放：与 `PathRunner`、`SubspaceModel` 同样的 Eigen 操作序列，热、冷的几何平均 |

输出：每个类别的估算、实测、比值，按项汇总（F1 F2 F3 C1 C2 C3 C0 G1 S1 A1 A2 G2 H1 V1 E M；G1 光滑部分组装、V1 真实能量与有效性、G2 梯度与重读、H1 链、E 精确项、M 映射与求解器胶水），实测份额 ≥ 3% 且估算 / 实测不在 [0.8, 1.25] 的项标出，最后是总覆盖率（估算总和 / 实测）。

**在 Mac 上要重新测的单价**（每台机器、每个场景都要）：

1. `run_replay.sh`：CHOLMOD 分解、符号分析、回代（热、冷）、约化矩阵的块核、小运算和路径内代数的操作序列、元素核（`l4d`，要盲选运行的 `elem` 导出）、宽相、碰撞集合、窄相证书遍历、TI 查询的全部重放、金丝雀。
2. bench 运行（每个场景、每种选择方式各一次，在 Mac 上跑导出构建）：问题侧各操作的价和 `S1.rep`。
3. 单价和计数都只和同一台机器上的参考比；Mac 的 Accelerate 与 MKL 末位不同，bowl 一类混沌场景要多个种子。
4. 导出运行本身也要在 Mac 上跑（计数、规模、`tiq_all.bin` 都是那台机器那次运行的）。

Linux 到 Mac 的搬法见第 4 节；类别模式不需要额外的东西。

### 覆盖率（11-23，`example/coverage_*.txt` 是完整输出，`unit_costs_pr_*.json` 是那次的单价）

被核对的运行是只开计数和类别计时的运行（`CL_NO_EXPORT=1`），单价来自导出运行的重放和同一场景的 bench 运行。估算 / 实测总和（实测 = 求解的线程 CPU 时间 − 钩子时间）：

| 场景 | 覆盖率 | 实测份额 ≥ 3% 且估算 / 实测不在 [0.8, 1.25] 的项 |
|---|---|---|
| mat3k_45 能量分项 c4R2 | 1.017 | 无（F1 1.00、C1 1.16、C2 0.93、C3 0.97、G1 0.97、G2 1.06、H1 0.98、V1 1.05） |
| mat3k_45 盲选 | 1.012 | 无（F3 0.90、C1 1.15、C2 0.87、G1 1.14、A1 1.02、H1 0.92、V1 0.89） |
| slide 前 20 步（摩擦） | 1.045 | 无（C2 占 77%，1.01：全部 TI 查询的重放） |
| bowl_19_it1000 能量分项 | 0.964 | A2 0.73（占 5.1%） |
| bowl_19_it1000 盲选 | 0.954 | A2 0.78（5.9%）、H1 0.78（4.5%） |

缺的部分：路径内的稠密代数（`P.run`、`A2.col`）在运行里比孤立重放慢约 1.4 倍（bowl 上 Q 是 n×k 的大矩阵，运行里的缓存状态比 96 MB 冲刷后的"冷"还差）；盲选 bowl 的 `NT.hessian_blocks`（按碰撞对加精确单元块数计价，估算低 30%）。其余类别都在区间里。
