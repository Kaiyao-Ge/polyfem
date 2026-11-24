# 导出树：为计数工具另编一份带导出钩子的 PolyFEM

计数工具不要求正式代码里有日志或导出。需要"规模类"数据（每次分解的 flops 与 lnz、每个接受点的块数、精确单元数和精确对数、每次 CCD 调用的候选数、交给 Tight Inclusion 的窄相查询）时，从正式代码另编一份，只加导出钩子。这里放的是做这一份所需的脚本和钩子补丁；复制出来的树和构建都在仓库外的工作目录 `$EXPORT_WORK`（默认 `~/nonsmooth_export`）里。导出构建不是正式代码，可以包含诊断。

## 目录

| 路径 | 内容 |
|---|---|
| `$EXPORT_WORK/pf_orig/`、`$EXPORT_WORK/ps_orig/` | 正式代码工作树的原样副本（polyfem 默认取本仓库，polysolve 默认取 `~/polysolve-pr`，用 `PF_SRC`、`PS_SRC` 改），不带 `.git`、构建目录和 `data` |
| `$EXPORT_WORK/pf/`、`$EXPORT_WORK/ps/` | 在副本上加了钩子的树，导出构建编译这两棵 |
| `patches/polyfem_export.patch`、`patches/polysolve_export.patch` | 钩子的改动，`diff -ruN pf_orig pf`、`diff -ruN ps_orig ps` |
| `sync.sh` | 正式代码更新后：重新复制到 `pf_orig/`、`ps_orig/`，再复制到 `pf/`、`ps/` 并 `patch -p1` 打钩子。不碰任何 git 仓库 |
| `make_patches.sh` | 改了 `pf/` 或 `ps/` 之后重新生成补丁 |
| `build.sh` | 配置（第一次）并编译 `build/PolyFEM_bin`（Release、Ninja、MISO 关、测试关，与正式构建 `build-blind` 同样的选项） |
| `$EXPORT_WORK/build/` | 构建目录 |

## 开关

两个环境变量，导出树不是正式代码，所以用环境变量。都不设时，钩子只是每处一次对静态标志的判断，求解的解和计数器与正式二进制逐位相同。设了以后，求解的轨迹也不变（钩子只读，不动数值）。

| 变量 | 作用 |
|---|---|
| `POLYFEM_CL_COUNTS=<目录>` | 写 `<目录>/cl_counts.jsonl`，每行一个 JSON：`solve`（每个求解一行）、`fact`（每次数值分解）、`outer`（每个外层）、`point`（每个接受点，约化矩阵每建一次一行）、`ccd`（每次宽相、窄相、碰撞集合构建） |
| `POLYFEM_CL_EXPORT=<目录>` | 写重放用的导出：`index.tsv` 和 `fact_*.bin`、`blocks_*.bin`、`elem_*.bin`、`surf_*.bin`、`tiq_*.bin`、`ccdmesh.bin`（格式见下） |

通常两个一起设（`<运行目录>` 与 `<运行目录>/export`）。导出选取在线做，磁盘上只留选中的集合：

- `fact`：每个外层的第一次 primary 分解，抽稀到 20 个（均匀，满了步长加倍）；所有失败的分解抽稀到 4 个；
- `blocks`、`elem`：接受点里均匀 8 个（含第一个）、最后一个、新块数最多的 2 个；
- `surf`：CCD 调用 12 个均匀 + 最后一个；碰撞集合构建 8 个均匀 + 最后一个（只导出用自己宽相的那种）；`tiq` 与它所在 CCD 调用的 `surf` 是同一个导出项；
- `ccdmesh`：一份。

## 类别计时（`cats` 行）与 bench

第二轮（全覆盖）加的钩子，都在导出树里，开关仍是 `POLYFEM_CL_COUNTS`（`POLYFEM_CL_NOTIMERS=1` 关掉计时）。

- **类别计时**：`polysolve/ClTime.hpp` 的 `cl::Scope("名字", 单位数)` 量它所在代码段的线程 CPU 时间，扣掉嵌套在里面的 Scope（独占时间），按名字累计时间、调用数和单位数。放在：求解器主循环（`S.*`）、`NLProblem` 的各个问题接口（`NL.*`、`NT.*`）、`FullNLProblem` 对每个形式的循环（`form.<名字>.value|grad|hess|hess.psd|step_valid|...`）、分解与回代（`F1`、`F2`、`F3`、链上的 `.chain`、选择里的 `F3.select`）、`BasisMatrix`、`PathRunner`（`P.run`、`P.eval`、`V1.true_energy`、`V1.jacobian_limit`、`G2.*`）、`SubspaceModel`（`A2.*`）、`ReducedMatrix`（`A1`）、选择（`S1`）、宽相（`form.barrier-contact.ls_begin`）、碰撞集合构建、碰撞网格位移（`CC.displace`）、有效性锚点（`NT.anchor`、`NT.jdets`）。每个求解结束时写一行 `{"t":"cats","step","total_cpu_ns","cats":{名字:{ns,calls,units}}}`。
- **`X.hook`**：导出钩子自己的时间（写行、写文件、记查询）。它和求解本身一起在 `total_cpu_ns` 里，估算对照时从总数里扣掉。
- **bench**：`POLYFEM_CL_BENCH=<时间步>`，在那个时间步的求解开始前，对问题侧的每个操作（`value`、`gradient`、`smooth_hessian`、`smooth_hessian_psd`、`nonsmooth_*`、有效性检查、`hessian`、`segment`、`solution_changed`、映射……）孤立地重复：热缓存（连续重复）和冷缓存（每次之前扫 48 MB）各一遍，写 `bench` 行（各类别的独占时间、调用数、单位数）。盲选运行里还把每次选择在同一数据上再重复 3 次（类别 `S1.rep`）。bench 会动问题的缓存，**要在单独的一次运行里做**，被核对的运行不开它。单价从 bench 行来，不从被核对的运行来。
- **`tiq_all.bin`**（CLTIQA01，在导出目录里）：每次窄相调用里真正交给 Tight Inclusion 的全部查询（预算 `POLYFEM_CL_TIQ_BUDGET`，默认 100 万条；用完后每 2、4、…次调用存一次，记录里带权重）。记录：`int64 seq, step, outer, nq, weight`，然后 nq 条 `int32 family, int32 nv, float64 tmax, float64 xa[12], float64 xb[12]`。Tight Inclusion 的代价是重尾的（滑动的平面接触里少数几次调用占了窄相的几乎全部时间），按查询的平均价或几次调用的抽样都会错一个数量级，所以重放整个运行的全部查询（`replay/cl_replay_ipc l4t`）。

## 计数行（cl_counts.jsonl）

- `{"t":"solve","step":i,...}`：一个求解（一次 minimize）开始，`step` 从 1 起，与 `sim.json` 里 `solver_info` 的次序对应。
- `{"t":"fact","step","outer","seq","kind","pattern_changed","n","nnz","fl","lnz","success","minor","fl_done","analyze_cpu_ns","factor_cpu_ns"}`：`kind` 是 `primary`（外层第一次）、`retry_proj`（整体投影后重试）、`repair`、`chain`（Newton 链）；`fl`、`lnz` 是 CHOLMOD 最近一次符号分析的浮点运算数和 L 的非零元数；失败时 `minor` 是停下的列，`fl_done` 是停下前做过的 flops；`pattern_changed` 表示这次分解前做了符号分析（F2）。
- `{"t":"outer","step","outer","inner_steps","inner_trunc","trunc":[guard,valid,ftb,ccd,jfloor,jbound],"projected","handed_off","lnz","d":{...}}`：每个外层一行，`d` 是这个外层里各计数器的增量（分解、回代、真实能量求值、非光滑状态重建、截断、重读、提升、选择）。
- `{"t":"point","step","outer","inner","point","n","k","nb","nb_by_size":{"6","9","12"},"exact_el","exact_pairs":{ee,fv,ev,vv,other},"a1_form","a1_nR","a1_nnzK","a1_written","a1_written_in","a1_new_blocks","a1_struct_built"}`：每个接受点（约化矩阵 A 每建一次）。`a1_form` 为 `current`（逐块）或 `merged_tau0`（合并）；不管实际用哪种，`a1_nR`、`a1_nnzK` 都按"这个外层第一个点的块结构 K_R"算，所以两种写法的估算都能做；`a1_new_blocks` 是不在 K_R 里的块数。
- `{"t":"ccd","kind":"broad|narrow|collisions",...}`：`broad` 是宽相（`Candidates::build`，每个路径段一次），`cand` 是各类候选数；`narrow` 的 `path` 是 `cert`（走证书的窄相：`cand` 是各类候选数，`ti` 是真正交给 Tight Inclusion 的查询数按类，`certified` 是被证书跳过的对数）或 `stock`（标准路径：全部候选交给 TI）；`collisions` 是碰撞集合构建，`cached` 表示用的是路径段的缓存候选（`ncand_in` 个），`cand` 是得到的碰撞数按类。

## 导出文件

`index.tsv` 的列和 `fact`、`blocks`、`elem`、`ccdmesh`、`surf` 的二进制格式见 `../counting/replay/cl_io.hpp`（重放程序直接读）。新增 `tiq_*.bin`（CLTIQ001）：

```
char[8]  magic = "CLTIQ001"
int64    nq, step, outer, point, ccd_max_iter
float64  ccd_tol, min_distance
nq 条，每条：  int32 family (0 vv, 1 ev, 2 ee, 3 fv)  int32 nv  float64 tmax  float64 xa[12]  float64 xb[12]
```

`xa`、`xb` 是模板的 `nv` 个顶点（3 个数一组，模板顺序）在路径段起点和终点的位置，`tmax` 是调用时的 earliest（证书窄相给每条 TI 查询的 tmax）。重放时对每条查询调 `stencil.ccd(xa, xb, toi, min_distance, tmax, ti)`，与求解器做的完全一样。`surf_N.bin` 与 `tiq_N.bin` 同一个导出项，`index.tsv` 里 surf 行的 `extra` 写 `narrow=cert`、`narrow=stock`（碰撞集合构建写 `build=own`）。

## 编译

```
tools/nonsmooth_counting/export/sync.sh && tools/nonsmooth_counting/export/build.sh      # ninja -j2；两个重文件先单独编；有内存上限的机器放在上限里跑
```

构建目录 `$EXPORT_WORK/build/`，二进制 `$EXPORT_WORK/build/PolyFEM_bin`。选项：Release、Ninja、`POLYFEM_WITH_MISO=OFF`、`POLYFEM_WITH_TESTS=OFF`、`POLYFEM_WITH_APP=OFF`、`CPM_polysolve_SOURCE=$EXPORT_WORK/ps`；设了 `CPM_SOURCE_CACHE` 就用它。并行数用 `NINJA_JOBS`（默认 2；在 7G 的内存上限下 `-j6` 会被限流到几乎不动）。

## 正式代码更新以后

```
tools/nonsmooth_counting/export/sync.sh        # 重新复制并打钩子；某个块打不上时脚本停下，rej 文件在 pf/、ps/ 里
# 打不上：在 $EXPORT_WORK/pf 或 $EXPORT_WORK/ps 里手改，然后 make_patches.sh 重新生成补丁，再 build.sh
```

## 核对

- 钩子关：与正式代码编出的二进制在 mat3k_45 单线程下逐位相同（`tools/nonsmooth_identity/nonsmooth_identity.py`：`sol.bin`、`energy.csv`、`sim.json` 的计数器）。
- 钩子开：同一二进制开关前后 `sol.bin` 逐位相同（mat3k_45 与 slide 前 20 步）。

