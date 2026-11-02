# 扑翼小样本异构任务元学习代码

本项目按 [方案.md](方案.md) 实现。主模型采用 Iwata 与 Kumagai 的 **Meta-learning from Tasks with Heterogeneous Attribute Spaces**；TabPFN 论文仅用于参考合成表格任务的设计思路。目标为 `X=[HWI, U, f] → T`、`X=[HWI, U, f] → L`，分别预测修正后的周期平均推力和升力。

代码可以单独作为 GitHub 仓库根目录上传；运行不依赖上一层的 Python 文件。真实数据不随代码打包，需要单独放到服务器并配置路径。`方案.md` 是研究方案，本 README 是操作说明。

## 1. 已实现的内容

- 主模型的初始列表示、支持样本编码、属性／响应关系更新、查询编码、响应预测；共享参数、平均池化及可变行／列／响应数的批处理 mask。
- DS、DS+FT、DS+MAML、NP、NP+FT、NP+MAML 对照；MAML 通过 `torch.func.functional_call` 实现可微内层更新和二阶元梯度，FT 不修改保存的共享权重。
- 13 种既有回归方法及 Mean、线性 Ridge／Lasso／Bayesian Ridge、线性 KRR、目标 MLP、固定平方指数 GP。可选外部 TabPFN 不属于相同预训练预算的核心对照。
- Dryad、ASCI、可选 DelFly 数据导入；通用 DAG、翼型、扑翼统计任务合成；任务与工况分区审计。
- 合成预训练、真实源数据混合元训练、固定源验证、早停、原子检查点和断点续训。
- 所有目标模型共用的外层 LOO、内层 3 折、少样本曲线、按 HWI／风速／频率整组留出、四种滤波以及周期质量敏感性分析。
- 新工况独立预测，可选择在新标签提供后计算指标。

合成数据是**自行实现的统计先验任务**，不是 CFD 或实测气动力；没有复制 TabPFN 未公开的预训练生成器。`mars`、`xgboost_lite` 是既有简化实现；`local_multilinear` 为局部加权多线性回归近似，不能称为严格规则网格插值。

## 2. 环境安装

推荐服务器 Linux、Python 3.11。项目支持 Python 3.11–3.13。CPU 可以运行；完整训练建议使用 NVIDIA GPU。训练设备 `auto` 优先 CUDA，否则使用 CPU；默认不会自动启用 MPS。

先安装 Miniconda 或 Anaconda，再从项目根目录创建名为 `wing-hml` 的 Conda 虚拟环境。若当前终端仍激活了旧的 `.venv`（提示符包含 `(.venv)`），先执行 `deactivate` 退出。

```bash
conda create -n wing-hml -c conda-forge python=3.11 pip -y
conda activate wing-hml
python -m pip install --upgrade pip
```

也可将上面的 `conda create` 命令替换为 `conda env create -f environment.yml`，然后执行相同的激活与安装步骤。`environment.yml` 只创建 Python 与 pip 基础环境；项目依赖在激活 `wing-hml` 后安装。若该环境已存在，直接执行 `conda activate wing-hml`，无需重复创建。

以下 CPU 与 NVIDIA 安装方式选择一种，所有命令均在已激活的 `wing-hml` 环境内执行。

CPU 环境安装（Linux）：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt 'pytest>=8,<9'
python -m pip install --no-deps -e .
```

NVIDIA 环境的安装示例（CUDA 12.8 wheel）：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-tested.txt 'pytest>=8,<9'
python -m pip install --no-deps -e .
```

CUDA wheel 要与服务器驱动兼容；可以按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择兼容的 wheel。Linux 驱动不由本项目安装。

安装后检查依赖并记录环境：

```bash
mkdir -p runs
python -m pip check
python -m wing_hml doctor --out runs/environment.json
```

NVIDIA 环境的 `doctor` 应显示 `cuda_available: true`，才表示当前 Python 可以使用 CUDA；CPU 环境显示 `false` 属于正常情况。

文件用途：`requirements.txt` 给出依赖版本范围，`requirements-tested.txt` 固定本地验证过的直接依赖版本，`requirements-dev.txt` 额外安装测试依赖。它们不是所有传递依赖的完整锁文件。每次服务器运行另存：

```bash
mkdir -p runs
python -m pip freeze > runs/server-pip-freeze.txt
python -m wing_hml doctor --out runs/server-environment.json
```

以后每次打开新终端，先执行 `conda activate wing-hml`，再运行训练或测试命令。所有命令均可写成 `wing-hml ...`；README 使用 `python -m wing_hml ...` 以明确 Python 环境。

## 3. 服务器数据与路径配置

原始源数据根目录应保持以下布局。可以只提供 ASCI 或 Dryad；没有任何可用源数据时 `prepare` 会报错。混合训练需要非空的真实源训练集和验证集。

```text
/srv/wing-data/
├── data_ASCI/
│   ├── GLASGOW/NACA0015/ 和 NACA0030/
│   ├── NREL/LS(1)0417MOD/、NACA4415/、S809/
│   ├── CENER/NACA64418/ 与 Selected_runs.xlsx
│   └── Oldenburg/Forces/、Pressures/ 与 Test-matrix_selected.xlsx
├── doi_10_5061_dryad_zgmsbccbs__v20210701/
│   └── flapping_wing_aerodynamics.mat
└── Data supporting the publication_ Delfly Flex_..._all/
    └── 原始 TXT／TDMS／MAT 文件（可选）
```

目标 CSV 至少含以下列：

```text
HWI,wind_speed_m_s,flapping_frequency_Hz,corrected_mean_thrust_N,corrected_mean_lift_N
```

`angle_deg`、`selection_status`、`wing_id` 可作为元数据。相同工况只能有一行；如果文件包含四种滤波，额外用 `filter_method` 区分，每种滤波仍为相同的 64 个工况。变化的迎角不能隐含混入当前固定迎角任务。

复制 `configs/server.example.yaml` 为 `configs/server.local.yaml`，填写实际的 `raw_data` 与 `target`。示例中的 `/srv/...` 是需要替换的服务器路径。`*.local.yaml` 已加入 Git 忽略规则。配置中的相对路径统一相对于**项目根目录**解析；`extends` 相对于该 YAML 所在目录解析。CLI 显式的相对路径也相对于项目根目录解析。

不指定服务器配置时使用 `configs/default.yaml`，其原始数据路径是本地目录布局的相对路径。可用重复的 `--set` 覆盖参数，例如 `--set device=cuda --set training.batch_size=16`。修改合成器、分区或实验配置时应选择新的输出目录，已有数据目录不会被隐式覆盖。

## 4. 先验证环境与短流程

没有真实数据也能运行单元测试及几步合成训练：

```bash
python -m pytest --basetemp runs/test-tmp
python -m wing_hml synthesize --config configs/smoke.yaml
python -m wing_hml train --config configs/smoke.yaml --stage synthetic --method hml --seed 42 --out runs/demo_synthetic/hml/seed_42
```

有真实源数据及目标 CSV 后，完整短流程为：

```bash
python -m wing_hml prepare --config configs/smoke.yaml --data-root /srv/wing-data
python -m wing_hml splits --config configs/smoke.yaml --target /srv/wing-target/mean_forces.csv
python -m wing_hml train --config configs/smoke.yaml --method hml --seed 42
python -m wing_hml evaluate --config configs/smoke.yaml --target /srv/wing-target/mean_forces.csv --models hml,mean --seeds 42 --protocols loo
```

如果之前已经生成 `smoke` 合成任务，继续使用即可；若未生成，先执行上一组 `synthesize`。短配置只有少量任务和更新，用于检查运行链路，**不能用来报告研究性能**。测试生成的数据、检查点和结果都在被 Git 忽略的目录中。

## 5. 正式准备数据

后续示例使用已编辑的 `configs/server.local.yaml`。

```bash
python -m wing_hml prepare --config configs/server.local.yaml
python -m wing_hml synthesize --config configs/server.local.yaml
python -m wing_hml splits --config configs/server.local.yaml
python -m wing_hml audit --config configs/server.local.yaml --out runs/audit.json
```

`prepare` 只读取原始文件，在 `data_work/real` 写出每个标量响应的任务 `.npz`、`manifest.json` 和 `summary.json`。ASCI 默认整组保留 S809 作验证、CENER NACA64418 作测试；其他翼型作训练。Dryad 以轨迹编号固定划分约 80%／10%／10%，各力与力矩输出共用轨迹分区。该划分的含义分别是跨翼型检验与同装置新运动检验，不能混称跨装置验证。

ASCI 的静态 `Time=NaN` 曲线会保留。动态输入使用向后差分、由输入运动估计的相位及可匹配的 CENER 工况；测力 Cd 与测压 Cdp 分开。动态轨迹也会汇总为每条记录的平均系数任务，其均值是原文件整个有效记录均值，不能宣称已做本机的“稳定 15 周期”筛选。Dryad 默认将每条运动轨迹转为一行运动摘要，按首末扑动峰之间的完整周期求平均；无法取得两峰时记录使用整段的回退方式，保留所用区间。不会使用原 PRSSM／LASSO 预测值充当实测标签。

可选源任务通过配置开启（应使用新的 `paths.prepared`）：

```bash
python -m wing_hml prepare --config configs/server.local.yaml --out data_work/real_with_aux --set prepare.include_dryad_instantaneous=true --set prepare.include_delfly_tdms=true --set prepare.include_flight=true
```

DelFly 周期均值只有每机体 5 个机动条件，默认不参与主训练；若开启 `prepare.include_delfly_summary=true`，还需把 `prepare.min_rows` 设为 5 或更小。TDMS 窗口任务与飞行动力学任务标为辅助任务；后者预测未来姿态、角速度或加速度，不能命名为升推力。

`synthesize` 默认生成 80,000 训练、10,000 验证、10,000 测试任务的**惰性规格**。每个任务的函数、DAG、参数、随机种子和排列都保存在 `tasks.jsonl`，训练时按需生成并缓存，避免存放 10 万个小文件。只有加 `--materialize` 才额外保存合成表格。通用／翼型／扑翼比例为 30%／30%／40%；测试任务一半为保留机制／范围的 OOD 任务。合成参数来自预设配置，没有用目标 64 组标签拟合生成器。

`splits` 保存所有目标方法共用的条件索引及内层分折。默认核对恰有 64 行；质量敏感性使用 `--quality-only`，自定义测试数据才使用 `--expected-rows 0`。四种滤波必须有相同的工况集合，否则需分别生成划分文件。

## 6. 元训练与断点续训

主模型单个种子：

```bash
python -m wing_hml train --config configs/server.local.yaml --method hml --seed 42
```

默认先进行最多 100,000 次合成更新，再进行最多 20,000 次合成／真实源任务混合更新。学习率分别为 0.001、0.0001；批大小 32，每 500 次验证，连续 20 次未改善时早停。混合阶段按概率 0.5 选真实源任务，真实来源内部均衡采样。各支持集与查询集互斥，标准化只使用支持集；验证 episode 固定并保存。训练命令不读取目标 CSV。

训练输出：

```text
runs/train/hml/seed_42/
├── run.json                    # 配置、源数据摘要、依赖、代码摘要、参数量
├── validation_episodes.json    # 固定源验证任务与行索引
├── history.jsonl
├── synthetic_best.pt
├── mixed_best.pt
├── best.pt                     # 当前最终阶段最佳模型，供 evaluate 使用
├── last.pt                     # 最近验证点的模型、优化器及随机状态
└── completed.json
```

中断后使用相同配置：

```bash
python -m wing_hml train --config configs/server.local.yaml --method hml --seed 42 --resume
```

`--auto-resume` 会在 `last.pt` 存在时恢复。恢复会检查方法、种子、网络、episode、训练参数及源数据摘要；允许增加更新次数上限，其他配置变更应建立新运行。检查点包含 Python 对象与优化器状态，只加载自己生成或可信的文件。不同硬件／CUDA 版本间不保证逐位一致。

只做合成阶段：

```bash
python -m wing_hml train --config configs/server.local.yaml --method hml --seed 42 --stage synthetic --out runs/synthetic_only/train/hml/seed_42
```

从该检查点开始一个新的混合阶段：

```bash
python -m wing_hml train --config configs/server.local.yaml --method hml --seed 42 --stage mixed --init-checkpoint runs/synthetic_only/train/hml/seed_42/best.pt --out runs/mixed_from_source/train/hml/seed_42
```

训练核心对照时，方法名是 `hml, ds, np, ds_maml, np_maml`。`ds_ft`、`np_ft` 分别复用 DS、NP 的源检查点，只在目标支持集上增加微调，不另外预训练一套权重。MAML 是二阶实现，训练会比主模型慢。

若要顺序训练全部方法和 5 个种子，脚本会准备缺失的数据并自动恢复已有运行：

```bash
export DATA_ROOT=/srv/wing-data
export TARGET_CSV=/srv/wing-target/mean_forces.csv
export CONFIG=configs/server.local.yaml
bash scripts/run_server.sh
```

只训练主模型一个种子可以设置 `METHODS='hml' SEEDS='42' bash scripts/run_server.sh`。脚本仅训练，不自动开始目标排名。Slurm 示例见 `scripts/train_slurm.sh`；先按集群修改资源配置，并在项目根目录执行 `mkdir -p runs/slurm`，然后例如 `METHOD=hml SEED=42 CONFIG=configs/server.local.yaml sbatch scripts/train_slurm.sh`。此项目不会自动提交远程作业。

## 7. 统一目标评估

仅比较传统 13 种模型，无需神经网络检查点：

```bash
python -m wing_hml evaluate --config configs/server.local.yaml --models historical --protocols loo --out runs/historical_loo
```

训练好全部源方法与种子后，运行核心比较：

```bash
python -m wing_hml evaluate --config configs/server.local.yaml --models neural,historical,simple --protocols loo
```

只训练过 HML 的种子 42 时明确指定：

```bash
python -m wing_hml evaluate --config configs/server.local.yaml --models hml,historical,mean --seeds 42 --protocols loo --out runs/hml_vs_baselines_42
```

主结果每轮 63 组支持、1 组留出，共 64 轮。需选参的模型只在 63 组内执行同一 3 折划分；固定配置跳过搜索。HML 从支持集推断列表示，不做目标梯度更新。FT／MAML 每次验证与测试都从原源权重开始，不沿用上一折的适应权重。K=3、5 的少样本实验按方案采用固定配置，不作目标内层选参。

GPR 选择平方指数、Matérn 3/2 或 5/2，噪声为训练目标标准差 5%，长度尺度保持既有固定设定；RBF 岭系数候选为 `1e-6,1e-3`；SVR 的 C 为 `10,100`、epsilon 为 `0.01,0.05`；随机森林 100 棵树、无最大深度限制，最小叶样本数为 `1,3,5,8`。全部固定参数及候选见 `baselines.py`，都在相同折内重新拟合预处理。

神经方法默认从 `runs/train/<base_method>/seed_<seed>/best.pt` 读取检查点。`--checkpoint-root` 可替换根目录；单模型也可用 `--checkpoint hml=/path/to/best.pt`。检查点种子与模型必须匹配，缺失会在目标评估开始前报错。源权重检查点的选择只使用源验证，不能根据 LOO 排名反选。

整组留出、少样本曲线：

```bash
python -m wing_hml evaluate --config configs/server.local.yaml --models hml,historical,mean --seeds 42 --protocols leave_hwi,leave_wind,leave_frequency --out runs/group_holdouts
python -m wing_hml evaluate --config configs/server.local.yaml --models hml,historical,mean --seeds 42 --protocols fewshot_3,fewshot_5,fewshot_8,fewshot_16,fewshot_32,fewshot_48 --out runs/fewshot
```

周期质量敏感性：

```bash
python -m wing_hml splits --config configs/server.local.yaml --quality-only --out data_work/splits_ok.json
python -m wing_hml evaluate --config configs/server.local.yaml --quality-only --splits data_work/splits_ok.json --models hml,historical,mean --seeds 42 --out runs/quality_ok
```

四种滤波：用 256 行四滤波 CSV 替换 `--target`，保留同一个 64 工况划分文件，并传入 `--filter all`。每种滤波单独训练目标支持集或条件推断、单独输出，绝不将 256 行混成一个目标数据集。

## 8. 结果、恢复与源测试

每次评估的输出在 `--out` 下按滤波分目录；普通单表目录名为 `default`，其他滤波使用稳定摘要，真实名称写入 `run.json`：

| 文件 | 内容 |
|---|---|
| `run.json` | 配置、条件划分、输入文件和检查点摘要、依赖及代码摘要 |
| `predictions.jsonl` | 可恢复的逐点日志 |
| `predictions.csv` | 每个工况的实测值、预测值、参数、种子及折编号 |
| `metrics.csv` | 每模型／目标／种子／协议的完整预测指标与覆盖率 |
| `summary.csv` | 已完成方法的 RMSE、MAE、R² 均值与种子标准差、RMSE 排名 |
| `fold_metrics.csv` | 分折指标；单点折的 R² 留空 |
| `completion.json` | 完成情况及局部运行标记 |

LOO 和整组留出的指标由所有留出点合并计算，不能平均单点 R²。少样本曲线先计算每次支持集划分的指标，再汇总划分，避免把重复出现的查询当独立实验。确定性模型只计算第一个种子；随机森林、浅树 boosting、MLP 和神经模型按指定种子重复。

中断后用原命令加 `--resume`。也可分段使用 `--fold-start 0 --fold-stop 16`，然后以相同方法、种子和协议执行 `--resume --fold-start 16`。不要让两个进程同时写同一个输出目录。只完成部分工况时不生成正式排名。修改目标标签、划分、代码或检查点后，恢复会拒绝签名不一致；应新建输出目录。默认遇到模型异常直接停止，排查用的 `--allow-failures` 会保留失败记录，但不会把不完整方法排为成功。

源任务测试分别运行，报告标准化 MSE，不能与目标 N 单位的误差混表：

```bash
python -m wing_hml meta-evaluate --config configs/server.local.yaml --checkpoint runs/train/hml/seed_42/best.pt --corpus data_work/synthetic --partition test --episodes 256 --out runs/meta_test_synthetic.json
python -m wing_hml meta-evaluate --config configs/server.local.yaml --checkpoint runs/train/hml/seed_42/best.pt --corpus data_work/real --partition test --episodes 256 --out runs/meta_test_real.json
```

## 9. 数据与模型消融

通用合成任务消融配置：

```bash
python -m wing_hml synthesize --config configs/generic_only.yaml
python -m wing_hml train --config configs/generic_only.yaml --method hml --seed 42 --stage synthetic
```

全部合成但不使用真实源任务：按第 6 节仅运行合成阶段。全部合成加真实源任务：运行默认两阶段。关系编码消融：新建输出目录并加 `--set model.relationship_update=false`；来源抽样消融加 `--set training.balanced_real=false`。所有变体应使用同一目标划分；源验证分别选检查点。不得将短配置训练结果混入正式对比。

## 10. 8 组新工况独立预测

当前独立实测数据仍缺失，代码已预留入口。查询 CSV 需要三个输入列，若同时提供两个实测力列，代码在预测完成后才用于评分：

```text
HWI,wind_speed_m_s,flapping_frequency_Hz,angle_deg
27.5,3,2.5,5
42.5,5,3.5,5
```

```bash
python -m wing_hml predict --config configs/server.local.yaml --query /srv/wing-target/new_conditions.csv --models hml,historical,mean --seeds 42 --out runs/independent
```

所有方法用旧 64 组选择参数并拟合／构造支持集，预测同一新工况。新工况若与旧工况重复或迎角变化，当前固定任务入口会报错。没有新实测标签时只输出预测，不生成实验验证指标。

## 11. 上传到 GitHub 与本地验证范围

建议以此目录作为仓库根目录。提交 `src/`、`configs/`、`scripts/`、`tests/`、环境文件、`README.md` 和 `方案.md`。`.gitignore` 排除了 `.venv/`、缓存、`data_work/`、`data_raw/`、`runs/` 及模型／原始 MAT、TDMS 文件；服务器需要单独准备原始数据和目标 CSV。GitHub Actions 的 CPU 测试配置位于 `.github/workflows/tests.yml`，它只测试代码，不启动正式训练。

本地验证使用 macOS ARM、Python 3.13、PyTorch 2.9.1 CPU；包括公式／不变性、二阶元梯度、支持集标准化、源分区、全部既有基线、精确续训和小规模链路。NVIDIA GPU、Slurm 以及完整 10 万任务长训练需在实际服务器上执行；仓库不包含已完成正式训练的权重或研究性能结论。

参考：[主方法](https://papers.nips.cc/paper_files/paper/2020/hash/438124b4c06f3a5caffab2c07863b617-Abstract.html)、[补充材料](https://papers.nips.cc/paper_files/paper/2020/file/438124b4c06f3a5caffab2c07863b617-Supplemental.pdf)、[合成任务参考论文](https://www.nature.com/articles/s41586-024-08328-6)。
