[readme_md.md](https://github.com/user-attachments/files/32639015/readme_md.md)
# TargetCAR-VLP：基于生成式人工智能的 CD8 靶向精准递送类病毒颗粒（VLP）从头计算设计与转化验证

## 1. 项目背景与技术路线设计闭环

### 1.1 体内 CAR-T 原位编程瓶颈与 VLP 载体优势
体外嵌合抗原受体 T 细胞（Ex vivo CAR-T）疗法受制于制备周期长、工艺成本高、细胞体外扩增耗竭以及患者清淋化疗预处理对骨髓微环境的破坏。实现体内原位精准转导是下一代免疫细胞治疗的核心演进方向。在基因与核酸递送载体选择中，当前主流方案面临多重生物物理瓶颈：
- **AAV 与慢病毒载体**：病毒衣壳天然具有强肝脏趋向性，慢病毒存在潜在随机整合致突变风险，且多次注射极易诱发体内针对病毒骨架的中和抗体反应；
- **脂质纳米颗粒（LNP）**：主要依赖载脂蛋白 E（ApoE）介导途径被动富集于肝实质细胞，即使表面功能化修饰，仍面临网状内皮系统（RES）清除、体内非特异性吸附及内涵体逃逸效率低下等多重障碍。

类病毒颗粒（Virus-like Particle, VLP）兼具假病毒的高效膜融合能力与非病毒载体的安全性：
1. **无自主复制基因组**：由纯结构蛋白自组装而成，杜绝基因组插入致突变风险；
2. **结构均一与渗透性强**：具有严格对称的球形多聚体外壳，流体动力学粒径高度均一（25–40 nm），组织渗透性极佳；
3. **多价展示与电荷可塑**：衣壳外表面可高密度、几何规则化展示特异性靶向元件，内表面可通过电荷工程进行物理化学性质的定制化调控。

### 1.2 分子尺度三大生物物理挑战
1. **CD8 亚基特异识别与非破坏性结合**：靶向弹头必须高特异性结合 CD8α 胞外 IgV 样结构域，同时避开阻断 MHC-I 类分子与 T 细胞受体（TCR）的天然接触界面，防止非特异性触发 T 细胞早期衰竭或过早激活；
2. **空间位阻与衣壳自组装兼容性**：天然单抗（IgG，~150 kDa）或单链抗体（scFv，~25 kDa）分子量庞大、构象柔性高，若高密度偶联于 VLP 衣壳表面，易破坏亚基间相互作用界面导致组装失败。亟须从头设计分子量小于 10 kDa、高热稳定性的微型结合蛋白（Mini-binder）；
3. **长链核酸货包超大静电排斥**：递送货包 CAR-mRNA（通常 1.5–3.0 kb）磷酸二酯骨架呈现高密度负电荷，天然蛋白空腔静电排斥剧烈。需在纳米外壳内部构建定向强正电荷微环境，实现长链核酸货包的自发凝缩与高密度封装。

### 1.3 “三位一体”计算设计闭环
<img width="2816" height="1536" alt="流程图" src="https://github.com/user-attachments/assets/c8058a04-d52a-4965-bc90-300fcea4a610" />


## 2. 预训练权重来源、模型搭建与跨环境解耦调度架构

依据大赛代码提交规范，**本项目全流程计算模型均直接调用官方开源代码仓库与公开预训练权重，未在本地开展模型训练或微调**。核心研发工作集中于靶向功能表位引导势能设计、几何正交骨架探索、序列防聚集负向偏置约束及三级数据质检漏斗构建。

### 2.1 硬件与集群基准环境
- **集群节点**：登录管理节点 `hpc_login01` / GPU 计算节点 `gpu01`
- **GPU 硬件**：4 × NVIDIA Tesla V100S-PCIE-32GB（Volta 架构，单卡配备 32GB HBM2 显存）
- **宿主显卡驱动**：NVIDIA Driver Version: 455.23.05（宿主驱动最高支持 CUDA 11.1 运行时）
- **作业调度系统**：IBM Spectrum LSF 10.1（提供 `bsub`, `bjobs`, `bkill` 指令集）

### 2.2 三大开源模型环境搭建与预训练权重获取途径
针对各模型底层深度学习框架与 CUDA 版本的依赖冲突，在集群上构建了 3 个相互独立的 Conda 虚拟环境：

1. **骨架生成环境（`rfdiffusion`）**：
   - **源码来源**：Baker Laboratory 官方开源代码库（`RosettaCommons/RFdiffusion`）
   - **权重获取**：通过官方 AWS S3 存储通道拉取官方发布的预训练模型权重 `ActiveSite_ckpt.pt` 与 `Base_ckpt.pt`（无条件与条件去噪扩散核心参数，体积约 2.4 GB）
   - **运行依赖**：Python 3.9, PyTorch 1.12.1+cu116, DGL 0.9.x

2. **序列逆折叠环境（`proteinmpnn`）**：
   - **源码来源**：Dauparas et al. 官方开源代码库（`dauparas/ProteinMPNN`）
   - **权重获取**：直接调用仓库自带的官方发布版预训练权重 `vanilla_model_weights/v_48_020.pt`（基于高分辨率 PDB 晶体结构训练的经典 48 空间邻域、0.20 Å 主链高斯加噪模型，体积约 119.9 MB）
   - **运行依赖**：Python 3.10, PyTorch 2.x（采用 cu118 独立纯张量核心，规避计算机视觉图像库编译冲突）, Biopython 1.88, NumPy 2.2.6, SciPy 1.15.3

3. **终审评估环境（`boltz2`，兼项目主复核环境）**：
   - **源码来源**：Valence Labs / MIT 官方开源套件（`HannesStark/boltz`）
   - **权重获取**：通过官方分发通道获取 Boltz-2 全原子多模态共折叠预训练参数包与亲和力回归预测头权重，本地保存在缓存路径中
   - **运行依赖**：Python 3.10.14, PyTorch 2.4.1+cu118, PyTorch-Lightning 2.5.0, Biopython 1.88, RDKit, matplotlib 3.10.9, pandas 2.2.0
   - **算子兼容与系统级补丁**：
     - *TLS 空间溢出修复*：Linux 动态链接器为动态库预留che的静态 TLS 存储有限，通过设置 `export LD_PRELOAD=/usr/lib64/libgomp.so.1:$LD_PRELOAD` 强制优先加载系统级 OpenMP 库，彻底解决 `dlopen: cannot load any more object with static TLS` 故障
     - *三角注意力原生回退*：Tesla V100S（Volta 架构）缺少现代 Triton 算子支持，运行推理时显式指定 `--no_trifast` 参数，平稳回退至 PyTorch 原生三角注意力实现
     - *显存段扩展防碎片*：配置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低高负荷下显存瞬时碎片峰值

### 2.3 跨环境解耦调度架构与显存 100% 释放机制
为杜绝多模型串联运行造成的显存碎片累积与爆显存（CUDA OOM）风险，计算管线在软件架构上执行严格的进程隔离规范：
- 由主控调度器通过独立子进程分别调用各虚拟环境下的独立 Python 解释器（`bin/python`）执行分段任务；
- 单阶段执行完毕后，子进程彻底退出，Linux 操作系统强制回收全部 GPU 显存至 0 MB，保障整个流水线在单卡上连续稳定运行。

---

## 3. 计算设计与验证全流程详细解析
### 3.1 阶段一：RFdiffusion 骨架从头连续扩散生成（01_diffusion）
- **流形扩散机理**：将多肽主链抽象为李群流形 $SE(3) = \mathbb{R}^3 \times SO(3)$ 上的随机微分方程（SDE）加噪与去噪过程。平移运动遵循方差保持型（VP）SDE，旋转扩散在李代数 $\mathfrak{so}(3)$ 上注入高斯微元。RoseTTAFold 三轨网络结合不变点注意力（IPA）直接预测无噪声构象闭式得分函数；
- **受体定义与热点引导**：受体采用人源 CD8α 胞外区晶体结构（PDB ID: 1CD8，Chain A）。选定远离 MHC-I 结合区的外露功能表位（Ser34 至 Phe107 核心功能区），引入界面接触势能函数：

<img width="627" height="103" alt="截屏2026-09-25 13 38 40" src="https://github.com/user-attachments/assets/10f5cd75-5e87-4803-93d0-a7f46ca89451" />
  强制新生骨架主链原子向预设表位贴合；
- **双构架拓扑探索**：
  - *Class I 构架*：设定生成长度 110–120 aa，定向引导为类 IgV 单域抗体（纳米抗体）的扁平 $\beta$-三明治折叠框架；
  - *Class II 构架*：配置 `contigmap.contigs=[A1-114/0 65-85]`，在固定 CD8α A 链 1–114 位条件下，从头生成 65–85 aa 的超紧凑 3-Helix Bundle 螺旋束；
- **阶段产出与质检**：生成数十个仅含 N, Cα, C, O 坐标的主链 PDB 文件。Top 候选二级结构比例 $(\alpha + \beta) \ge 60\%$，排除了柔性 Loop。在受体对齐下，Class I 与 Class II 骨架空间位姿 RMSD 达 22.4 Å，证实生成了占据不同子口袋的正交解。

### 3.2 阶段二：ProteinMPNN 序列几何逆折叠（02_mpnn）
- **几何图不变性提取**：构建空间残基 $k\text{-NN}$ 邻近图（$k=48$），提取 25 种重原子间距高斯径向基函数（RBF）展开特征与局部旋转四元数，严格保证旋转平移不变性；
- **约束偏置注入**：
  - 完全冻结受体 CD8α（Chain A）序列与空间坐标；
  - 解码温度缩放：界面结合区域设置中度熵采样 $T = 0.25$，产生多样化的侧链极性网络组合以供筛选；
  - 防聚集与抗氧化硬性偏置：注入半胱氨酸惩罚 `Cys = -10.0`，杜绝体外分子间二硫键错配聚集；注入甲硫氨酸惩罚 `Met = -5.0`，规避体内活性氧（ROS）氧化失活；
- **阶段产出与质检**：经自洽折叠初筛输出 28 个完整复合物 PDB 与 FASTA 序列文件，合格候选全部达到构象能量阱评分 $\le 1.10$，且严格保持 $0\text{ Cys}, 0\text{ Met}$。

### 3.3 阶段三：Boltz-2 全原子多模态共折叠与亲和力终审（03_boltz2）
- **物理打分机制**：依托 48 层 Pairformer 引擎在隐空间维持三角不等式几何闭环，在连续笛卡尔空间 $\mathbb{R}^{3 \times M}$ 对原子点云实施连续去噪重折叠，结合立体化学违背损失消除空间穿模。最终输出界面结合置信度（ipTM）、复合物折叠质量（Complex pLDDT）及整体置信度（pTM）；
- **阶段质检终审结果**：单体自折叠一致性 Cα-RMSD（vs RFdiffusion）$\le 1.2\text{ \AA}$，排除了 AI 幻觉构象；Complex pLDDT 达 83.00–89.51；全库仅 Rank 1（0.8383）与 Rank 2（0.8195）突破 0.80 超高置信阈值。

---

## 4. 全量 28 个样本预测打分总表（按 ipTM 降序排列）

全量打分数据归档于 `results/results.csv`：

| 排名 | 样本名称标识 | ipTM | pTM | Complex pLDDT | 拓扑分类与架构说明 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Rank 1** | `cd8_binder_0_seq_2` | **0.8383** | 0.8965 | 0.8951 | Class I：类 IgV 免疫球蛋白单域 (114 aa) |
| **Rank 2** | `cd8_binder_0_t0.3_s8` | **0.8195** | 0.9124 | 0.8300 | Class II：De Novo 三螺旋束 (65 aa) |
| **Rank 3** | `cd8_binder_1_t0.3_s3` | 0.7600 | 0.8492 | 0.8544 | Class II：De Novo 三螺旋束 (66 aa) |
| **Rank 4** | `cd8_binder_0_t0.3_s5` | 0.7304 | 0.8311 | 0.8830 | Class II：De Novo 三螺旋束 (65 aa) |
| **Rank 5** | `cd8_binder_1_seq_1` | 0.6907 | 0.8275 | 0.8563 | Class I：类 IgV 免疫球蛋白单域 (114 aa) |
| Rank 6 | `cd8_binder_1_t0.3_s1` | 0.6840 | 0.8367 | 0.8252 | Class II：De Novo 三螺旋束 |
| Rank 7 | `cd8_binder_1_t0.3_s2` | 0.6313 | 0.8235 | 0.8086 | Class II：De Novo 三螺旋束 |
| Rank 8 | `cd8_binder_1_t0.5_s12` | 0.6154 | 0.7851 | 0.9049 | Class II：De Novo 三螺旋束 |
| Rank 9 | `cd8_binder_1_seq_4` | 0.6118 | 0.7898 | 0.8617 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 10 | `cd8_binder_1_seq_2` | 0.6110 | 0.7890 | 0.8770 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 11 | `cd8_binder_1_t0.3_s4` | 0.5892 | 0.7728 | 0.8647 | Class II：De Novo 三螺旋束 |
| Rank 12 | `cd8_binder_0_seq_1` | 0.5765 | 0.7674 | 0.8689 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 13 | `cd8_binder_1_t0.3_s6` | 0.5468 | 0.7595 | 0.8631 | Class II：De Novo 三螺旋束 |
| Rank 14 | `cd8_binder_1_t0.3_s7` | 0.5403 | 0.7627 | 0.9109 | Class II：De Novo 三螺旋束 |
| Rank 15 | `cd8_binder_0_t0.3_s3` | 0.5203 | 0.7741 | 0.8958 | Class II：De Novo 三螺旋束 |
| Rank 16 | `cd8_binder_0_t0.5_s14` | 0.5083 | 0.7646 | 0.8806 | Class II：De Novo 三螺旋束 |
| Rank 17 | `cd8_binder_0_t0.3_s1` | 0.4620 | 0.7462 | 0.8525 | Class II：De Novo 三螺旋束 |
| Rank 18 | `cd8_binder_0_t0.3_s6` | 0.4583 | 0.7563 | 0.8884 | Class II：De Novo 三螺旋束 |
| Rank 19 | `cd8_binder_1_t0.3_s8` | 0.4439 | 0.7403 | 0.8585 | Class II：De Novo 三螺旋束 |
| Rank 20 | `cd8_binder_1_seq_3` | 0.4178 | 0.6866 | 0.8477 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 21 | `cd8_binder_0_seq_4` | 0.4060 | 0.6813 | 0.8707 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 22 | `cd8_binder_0_t0.3_s2` | 0.3480 | 0.7235 | 0.8498 | Class II：De Novo 三螺旋束 |
| Rank 23 | `cd8_binder_1_t0.5_s13` | 0.3174 | 0.7103 | 0.8353 | Class II：De Novo 三螺旋束 |
| Rank 24 | `cd8_binder_0_t0.3_s7` | 0.3027 | 0.7123 | 0.8623 | Class II：De Novo 三螺旋束 |
| Rank 25 | `cd8_binder_0_seq_3` | 0.2388 | 0.5949 | 0.8332 | Class I：类 IgV 免疫球蛋白单域 |
| Rank 26 | `cd8_binder_1_t0.3_s5` | 0.2207 | 0.6722 | 0.8471 | Class II：De Novo 三螺旋束 |
| Rank 27 | `cd8_binder_0_t0.3_s4` | 0.1396 | 0.6535 | 0.8514 | Class II：De Novo 三螺旋束 |
| Rank 28 | `cd8_binder_1_seq_5` | 0.1145 | 0.5820 | 0.8120 | Class I：类 IgV 免疫球蛋白单域 |

---

## 5. Top 5 候选定量理化特性与成药性评估

理化参数详见 `results/top_binders_physicochemical.csv`：

| 候选体编号 | 样本标识 | ipTM | 长度 (aa) | 分子量 (kDa) | 等电点 (pI) | 净电荷 (pH 7.4) | GRAVY (疏水性) | 芳香族占比 | 拓扑类别 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Rank 1** | `cd8_binder_0_seq_2` | **0.8383** | 114 | 12.06 | 4.98 | -1.76 | -0.10 | 8.8% | Class I (类 IgV 单域) |
| **Rank 2** | `cd8_binder_0_t0.3_s8` | **0.8195** | 65 | 7.16 | 4.80 | -6.32 | -0.90 | 0.0% | Class II (3-Helix 螺旋束) |
| **Rank 3** | `cd8_binder_1_t0.3_s3` | 0.7600 | 66 | 8.47 | 5.34 | -3.39 | -2.32 | 0.0% | Class II (3-Helix 螺旋束) |
| **Rank 4** | `cd8_binder_0_t0.3_s5` | 0.7304 | 65 | 7.61 | 5.38 | -2.43 | -1.30 | 0.0% | Class II (3-Helix 螺旋束) |
| **Rank 5** | `cd8_binder_1_seq_1` | 0.6907 | 114 | 12.19 | 5.00 | -2.73 | -0.12 | 7.9% | Class I (类 IgV 单域) |

### 5.1 Top 2 候选成熟肽完整氨基酸序列
- **Rank 1 成熟肽序列 (114 aa)**：
  `SFFSVSPLNTTYKLGETVVLSDTSLLPAASTGQQWWFQPSGLTTPPTFLASVDSSAVTYASGVDTSRISASRSGRTSTLTIKNLQPEDAGYYFAQQTSGGVTYSSDPILVKLPS`
- **Rank 2 成熟肽序列 (65 aa)**：
  `ETAAEARRRAEAEAAAAAAEEAARQQRLAAEKQKELQALEAEALKLLEKLKAEAEAEEREREALE`

### 5.2 成药性与 VLP 衣壳偶联适配性机制
1. **Rank 2 (cd8_binder_0_t0.3_s8) 的极低位阻优势**：分子量仅 7.16 kDa（显著小于 10 kDa 设计门槛），GRAVY 达到 -0.90，具有极佳的水相溶解度与单分散性。若将其高密度展示于 VLP 衣壳表面，几乎不会对二十面体亚基间相互作用产生空间位阻破坏；
2. **Rank 1 (cd8_binder_0_seq_2) 的典型球蛋白特征**：GRAVY 为 -0.10，分子量 12.06 kDa，芳香族残基占比 8.8%，在生理缓冲液中呈现弱负电荷（-1.76），具备优良的抗体样折叠稳定性与界面疏水咬合能力；
3. **Rank 3 的淘汰机理**：GRAVY 低达 -2.32，序列中包含大段连续的同种电荷串联片段（连续酸性区 EEEE 与连续碱性区 RRR），在体外重折叠时极易形成无规非特异性胶束，故排除出优先推进清单。

---

## 6. 结合表位残基微环境与相互作用力学解析

以受体 CD8α（Chain A）与 Binder（Chain B）之间任意重原子接触距离 $\le 4.0\text{ \AA}$ 进行界面指纹提取：
- **CD8 受体覆盖残基池**：共计 29 个残基（从 Leu8 延展至 Phe111）；
- **Rank 1 接触残基**：共 19 个残基；
- **Rank 2 接触残基**：共 24 个残基；
- **共有核心结合残基（14 个）**：`Ser34, Ser45, Pro46, Phe48, Tyr51, Lys58, Tyr91, Leu97, Ser100, Ile101, Met102, Phe104, His106, Phe107`。

### 结合模式物理机制对比
- **Rank 1（类 IgV 单域深插型）**：利用 Loop 区垂直伸入 CD8 结合裂隙，严密封闭 Gln38、Ala43、Ala44 和 Phe93。由 CD8 的 Phe48、Tyr51、Leu97、Phe104 与 Rank 1 的芳香侧链（Phe2、Phe75、Tyr78）构建致密的 $\pi\text{-}\pi$ 堆积与疏水嵌合；
- **Rank 2（3-Helix 螺旋束表面覆盖型）**：以圆柱螺旋侧面平铺贴附于 CD8 表位裂隙外侧。外侧密集的带负电谷氨酸（Glu）侧链羧基与 CD8 表面碱性残基（Lys58、His106）形成多对高能量盐桥静电吸引，实现非破坏性高亲和力结合。

---

## 7. 湿实验重组表达载体工程构建（pET-28a 系统）

针对 Rank 1 与 Rank 2 设计了面向 *E. coli* BL21(DE3) 表达系统的完整质粒构建方案，序列保存在 `results/cd8_binders_dna_orders.fasta`：
- **载体骨架**：pET-28a(+)
- **克隆位点**：5' 端 NdeI（`CATATG`，含起始 ATG），3' 端 XhoI（`CTCGAG`）
- **标签与酶切位点**：N 端 6×His-Tag + GS 连接子 + TEV 蛋白酶识别位点（`ENLYFQ/G`，酶切后彻底脱除标签恢复原始成熟肽，避免干扰结合）
- **翻译终止策略**：串联双重终止密码子（`TAATGA`）
- **保护碱基设计**：5' 端附加 `ggaattc`，3' 端附加 `accggg`

### 构建体详细生物物理参数
1. **Rank 1 构建体 (`pET28a_CD8_Binder_Rank1_IgLike`)**：
   - 含标签前体：136 aa，理论分子量 14.53 kDa，理论等电点 pI 5.79，理论摩尔消光系数 $\varepsilon_{280} = 29,910\text{ M}^{-1}\text{cm}^{-1}$，编码区 GC 含量 52.3%；
   - DNA 订单序列（5' → 3'，全长 428 bp）：
     `ggaattccatatgGGTAGCAGCCATCATCATCACCACCATAGCAGCGGCGAAAACCTGTATTTTCAGGGCAGCTTTTTTAGCGTTAGCCCGCTGAACACCACCTATAAACTGGGCGAAACCGTGGTTCTGAGCGATACCAGCCTGCTGCCGGCAGCAAGCaccggccagcagtggtggtttcaaccgagcggcctgaccaccccgccgacctttctggcaagcgtggatagcagcgcagttacctatgcaagcggcgtggataccagccgcattagcgcgagccgcagcggccgcaccagcaccctgaccattaaaaacctgcaaccggaagatgcaggctattattttgcacaacaaaccagcggcggcgttacctatagcagcgatccgattctggttaaactgccgagcTAATGActcgagaccggg`

2. **Rank 2 构建体 (`pET28a_CD8_Binder_Rank2_HelixBundle`)**：
   - 含标签前体：87 aa，理论分子量 9.64 kDa，理论等电点 pI 5.86，理论摩尔消光系数 $\varepsilon_{280} = 1,490\text{ M}^{-1}\text{cm}^{-1}$（切除标签后成熟肽需用 A205/BCA 定量），编码区 GC 含量 54.8%；
   - DNA 订单序列（5' → 3'，全长 281 bp）：
     `ggaattccatatgGGTAGCAGCCATCATCATCACCACCATAGCAGCGGCGAAAACCTGTATTTTCAGGGCGAAACCGCAGCAGAAGCACGCCGCCGTGCAGAAGCAGAAGCAGCAGCAGCAGCAGCAGAAGAAGCAGCACGCCAACAACGCCTGGCCGCAGAAAAACAGAAAGAACTGCAAGCGCTGGAAGCAGAAGCCCTGAAACTGCTGGAAAAACTGAAAGCAGAAGCCGAAGCAGAAGAACGCGAACGCGAAGCACTGGAGTAATGActcgagaccggg`

---

## 8. 规范目录树与文件清单

```text
project/
├── README.md                           # 本项目完整技术规范与复核指南
├── requirements.txt                    # 终审与复核环境依赖库版本锁定清单
├── predict.py                          # 自动化结果复核与候选清单校验脚本
├── models/
│   └── MODEL_CARD.md                   # 模型开源来源、权重版本与参数配置说明卡
├── src/
│   └── plot_results.py                 # 全景评估图谱与交互页面生成代码
├── results/
│   ├── results.csv                     # 官方标准格式：全量 28 个样本预测打分总表
│   ├── top_binders_physicochemical.csv # Top 候选理化性质与成药性评估表
│   ├── top2_selected_binders.fasta     # 优选成熟肽氨基酸序列
│   ├── cd8_binders_dna_orders.fasta    # 重组表达质粒优化 DNA 合成订单
│   ├── cd8_binders_evaluation.png      # 4 面板 300 DPI 出版级评估大图
│   ├── boltz2_ranking_chart.svg        # ipTM 梯度排序矢量图
│   ├── epitope_overlap_heatmap.svg     # 界面接触残基指纹热图
│   ├── structures/                     # Top 5 复合物标准三维 PDB 结构文件
│   │   ├── rank1_cd8_binder_0_seq_2_iptm0.8383.pdb
│   │   ├── rank2_cd8_binder_0_t0.3_s8_iptm0.8195.pdb
│   │   ├── rank3_cd8_binder_1_t0.3_s3_iptm0.76.pdb
│   │   ├── rank4_cd8_binder_0_t0.3_s5_iptm0.7304.pdb
│   │   └── rank5_cd8_binder_1_seq_1_iptm0.6907.pdb
│   ├── view_rank1.html                 # Rank 1 网页端 3D 构象交互文件
│   └── view_rank2.html                 # Rank 2 网页端 3D 构象交互文件
└── logs/                               # 采样记录与执行日志
    ├── inference_verification.log      # 预测复核程序输出日志
    └── sampling_parameters.log         # 采样温度与随机种子记录表
```

---

## 9. 运行与复核命令

在集群激活 `boltz2` 运行环境后，直接在 `project` 根目录执行以下命令：

```bash
# 1. 结果复核与候选清单校验
conda activate boltz2
python predict.py

# 2. 重新绘制全维度多面板评估图表
python src/plot_results.py
```
