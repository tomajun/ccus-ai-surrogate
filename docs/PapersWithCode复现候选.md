# Papers with Code 相近项目清单 + 复现可行性排序

> 目的：按用户要求去 Papers with Code（及 GitHub topic）找与本项目（CO₂ 地质封存轻量级深度学习代理模型 + 风险分级）相近的项目，评估**能不能真复现**，给出"先复现、再优化"的落地首选。
> 调研日期：2026-10-15。
> 配套文档：`docs/复现对标论文与计划.md`（原 benchmark 路线，本文是其 PwC 复核修正版）。

---

## 〇、先说结论（最重要）

**之前锁定的 Zingaro 2026 是完美 benchmark，但没公开代码，只能"方法学复现"（自己按它的设置重做）。如果你要"拉别人代码跑通"式复现，首选必须换成 `U-FNO (Wen 2022, github.com/gegewen/ufno)`**——它有完整 Python 代码 + Google Drive 公开数据集，且正好是本项目 P2（FNO）的升级版。

> **2026-10-29 实测修正**：原写"本科 CPU 笔记本就能跑"只对**推理**成立（单样本前向实测 9.0 s）。**训练不可行**——训练集 `sg_train_a.pt` ≈ 99.5 GB，本机磁盘/内存都放不下。复现方式已改为**下载官方预训练权重 + test 集推理**，见 `数据集与结题可行性审计.md` 第四节。

一句话路线：
- **想省力、稳、能交差** → 复现 **U-FNO**（有代码有数据）。
- **想硬核、覆盖五架构、写进申报书"国内外现状"** → 复现 **Zingaro 2026**（无代码，自己重做五架构对比）。
- **想在 UQ 上直接对标** → 参考 **Engineering Applications of AI 2026 deep ensembles**（含 ensemble-UQ，正好对 P4）。

---

## 一、候选排序表（按"能否真复现 × 本科可落地"）

| 排名 | 项目 / 论文 | 代码可得 | 数据可得 | 本科 CPU 可跑 | 与本项目管线对齐度 | 推荐度 |
|---|---|---|---|---|---|---|
| ★1 | **U-FNO** (Wen 2022, `gegewen/ufno`) | ✅ GitHub 完整 | ✅ Google Drive 4500 样本 | ✅（小网格） | 高（FNO 升级版，覆盖 P1→P2） | **首选·代码复现** |
| ★2 | **Zingaro 2026** (arXiv:2606.07215) | ❌ 无公开 repo | ⚠️ 自生成 | ✅ | 极高（五架构 benchmark，覆盖 P1/P2） | **首选·方法学复现** |
| 3 | **PhysicsNeMo reservoir_simulation** (`NVIDIA/physicsnemo`) | ✅ 官方示例 | ✅ 含 2D open CO₂ 数据集 | ⚠️ 需装重框架 | 高（U-FNO/Conv-FNO/UNet） | 工业级参照 |
| 4 | **Deep ensembles for CO₂ saturation** (Eng. Apps. AI 2026) | ⚠️ 未确认 repo | ⚠️ 317 样本 | ✅ | 高（U-Net ensemble UQ，对 P4） | UQ 对标 |
| 5 | **FNO4CO2** (`slimgroup/FNO4CO2`) | ✅ 但 Julia | ✅ Dropbox | ❌ 需 Julia+Devito | 中（Julia 门槛） | 不推荐本科 |
| ✗ | Stanford `operator_co2_flow` (Ju&Alahmed 2024) | ❌ **404 失效** | — | — | — | **已排除** |

> 待核实高价值线索：GitHub `co2-sequestration` topic 下有一个直接含 **FNO + conditional-diffusion (U-Net & DiT) + ensemble-UQ** 的开源 benchmark（Python），搜索时只看到 topic 描述、没拿到具体 repo 名。建议你本机在 GitHub 搜 `co2-sequestration` topic 或 `ensemble-UQ CO2 sequestration` 直接 clone——它若属实，是"FNO+UQ 一体"的最佳复现源。

---

## 二、★ 首选（代码复现）：U-FNO — Wen et al. 2022

**仓库**：`https://github.com/gegewen/ufno`
**论文**：Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., Benson, S. M. (2022). *U-FNO — An enhanced Fourier neural operator-based deep-learning model for multiphase flow.* Advances in Water Resources 163, 104180. DOI:10.1016/j.advwatres.2022.104180
**数据集**：Google Drive（Train n=4500 / Val n=500 / Test n=500，`.pt` 格式，输入=渗透率/孔隙度场，输出=CO₂ 气体饱和度 sg 与压力 dP）
**要求**：PyTorch 1.8.0（基础 API，新版 torch 多半兼容）

### 为什么它是首选
1. **有代码有数据**——这是 FNO+CCS 方向极少数"论文+代码+数据"三位一体的开源工作，避免了"论文好看但 clone 不到"的坑。
2. **正好是你的 P2 升级靶子**——你的 `src/models/fno_surrogate.py` 是标准 FNO；U-FNO = 在 Fourier 层上挂一条 mini U-Net 支路增强高频细节。复现它 = 直接验证"FNO → U-FNO 的增益"，和你 P1/P2 无缝衔接。
3. **轻量**——2D 径向网格，CPU 可训（论文用 GPU 但小网格 CPU 几小时能出）。
4. **数据集格式 `.pt`**——和你 `experiments/*.pt` / `models/*.pt` 同生态，好对接。
5. **被引极高**——Zingaro 2026、PhysicsNeMo 示例、Chandra 2025 都拿它当 baseline，复现它在文献网里站得住。

### 复现步骤（你本机执行，沙箱做不了：需 git + Google 账号 + 装 torch）
```bat
:: 1) 克隆代码
git clone https://github.com/gegewen/ufno C:\workplace\ccus-ai-surrogate\repro\ufno
cd C:\workplace\ccus-ai-surrogate\repro\ufno

:: 2) 下载数据集（Google Drive 链接在 README；需 Google 账号，免 NETL 墙）
::    https://drive.google.com/drive/folders/1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo
::    把 sg_train_a.pt / sg_train_u.pt 等放进仓库根目录

:: 3) 建独立环境（用你已装好的 3.13.12 托管 venv，避免重装 torch）
C:\workplace\ccus-ai-surrogate\.venv\Scripts\python.exe -m pip install notebook   :: 若要用 .ipynb
:: 或把 train_UFNO_gas_saturation.ipynb 导出为 .py 直接跑

:: 4) 跑气体饱和度训练（复现靶子）
C:\workplace\ccus-ai-surrogate\.venv\Scripts\python.exe train_ufno_gas_saturation.py
::   复现目标指标：论文 gas saturation 相对误差在低个位数 %、R²>0.99
```

### 在你复现基础上的"优化"叠加（= 你大创的增量创新）
| U-FNO 原论文做到了 | 你叠加（差异化，已有代码可直接接） |
|---|---|
| 点估计（单张饱和场预测） | **深度集成 UQ**（`src/train_uq.py` 已写）→ 给 ±置信区间，决策方知"有多不准" |
| 只预测饱和场/压力 | **通道专属泄漏风险三档**（`sim/risk_proxy.py` 已写）→ caprock/wellbore/fault/lateral + 受体汇总，论文无 |
| 随机异质场（无真实锚） | **知识引导合成**（NATCARB 先验已落地 `data/priors.json`）→ 参数分布真实可辩护 |

> 思路比喻：U-FNO 是"快且准的引擎"，你把"仪表盘（UQ）+ 安全气囊（风险分级）+ 真油箱（真实先验）"装上，就成了一辆比原论文更完整的车——这就是答辩时"为什么不直接用论文方法"的答案。

---

## 三、★ 首选（方法学复现）：Zingaro 2026

**论文**：Zingaro, G., Gracie, R., Leonenko, Y. (2026). *A Comparative Study of Deep Learning Models for Geological Carbon Sequestration.* arXiv:2606.07215.
**核心结论（你要复现的靶子）**：在 2D 单井 CO₂ 注入（各向异性异质渗透率/孔隙度场）上对比 U-Net/V-Net/TCN/FNO/U-FNO；**U-FNO 在 CO₂ 饱和场最准，FNO 在压力场最准**；性能强依赖底层 PDE 类型（饱和=双曲、压力=椭圆）。

### 为什么仍重要
- 它是"哪个架构最好"的权威 benchmark，和你 P1(CNN)/P2(FNO) 完全同构，写进申报书"国内外研究现状"分量很重。
- 已被 Pith 机器审稿，方法透明、结论有第三方质疑记录（可引用其局限做自己创新铺垫）。

### 复现方式（无代码，自己重做）
- 用你现有框架按它的架构清单训至少 U-Net/CNN vs FNO vs U-FNO，产出 **baseline 对比图 + 误差表**。
- 严格对齐它的设置需把 `sim/synthetic_field.py` 升级为"异质渗透率场→饱和场"（阶段 B，约 1–2 周），详见 `docs/复现对标论文与计划.md` 阶段 B。
- 逐指标对齐它的数字较难（数据不同），但"FNO 类优于 CNN、U-FNO 优于 FNO"这一大类结论可交叉验证。

---

## 四、工业级参照：NVIDIA PhysicsNeMo reservoir_simulation

**仓库**：`https://github.com/NVIDIA/physicsnemo`（示例路径 `examples/reservoir_simulation/DeepONet/`）
**内容**：U-FNO / Conv-FNO / UNet + physics-informed losses + 分布式训练；EarthDoc 2026 论文《Comparative study on Neural Operators for Reservoir Simulation》明确"全部代码与数据集通过 PhysicsNeMo 开源"，含 **2D open-source CO₂ sequestration dataset（24 时间步）**。
**定位**：最"工业级"、最完整，但框架重（需装 PhysicsNeMo + 部分需 GPU），本科首次接触成本高。建议作为"方法验证黄金标准"而非首选复现源；若你后续想做 3D/工业级演示再切过来。

---

## 五、UQ 直接对标：Deep ensembles for CO₂ saturation (Eng. Apps. AI 2026)

**论文**：Vo Thanh H., Min B. (2026). *From digital rock to geological carbon storage decisions: Time-Resolved CO₂ saturation prediction with deep ensembles.* Engineering Applications of Artificial Intelligence 182, 116165.
**内容**：对比 DeepCNN/U-Net/ResU-Net/FNO2D 四家族预测孔隙尺度 CO₂ 饱和；**bootstrapped U-Net ensemble 提供空间不确定性图**——但论文自己承认其 80% 区间"under-dispersed"（欠分散），ensemble spread 当定位诊断而非校准区间。
**价值**：正好对你 P4 深度集成 UQ，可作为"interval calibration（区间校准）"这一短板的对照文献——你的 PICP/MPIW 指标能直接和它对话。⚠️ 代码 repo 未确认，先当文献对标。

---

## 六、已排除项（避免你踩坑）

- **Stanford `operator_co2_flow` (Ju&Alahmed 2024)**：论文声称代码在 `github.com/IsaacJu-debug/operator_co2_flow`，**实测 404 失效**（私有/删除）。作者另一仓库 `gnn_flow` 是 GNN 路线，非 CNN/FNO。**不可作为代码复现源。**
- **FNO4CO2 (`slimgroup/FNO4CO2`)**：Julia 语言 + Devito 波场模拟，复现门槛高，本科不推荐（除非你熟 Julia）。
- **Kimberlina / EDX 数据线**：已确认 EDX 需 `.GOV/.MIL` 邮箱或 NETL POC 担保，本科不可达，相关复现线全废（见 `docs/CCUS数据锚点验证计划.md`）。

---

## 七、"先复现再优化"落地路线（结合你现有管线）

```
阶段 A（代码复现·省力）             阶段 B（方法学 benchmark·硬核）        阶段 C（优化·你的增量）
─────────────────────             ─────────────────────────            ─────────────────────
clone gegewen/ufno                按 Zingaro 设置重做五架构对比          在 A/B 训好的 surrogate 上：
↓ 下 Google Drive 数据集           ↓ 升级 synthetic_field 为异质场       + 深度集成 UQ (train_uq.py)
↓ 跑 train_ufno_gas_saturation     ↓ 产出 baseline 对比图+误差表          + 风险三档 (risk_proxy.py)
→ 复现 RMSE/R² 指标                → 交叉验证"FNO类> CNN"                + 知识引导合成 (priors.json)
（已验证 FNO 骨架在你管线跑通）     （覆盖 P1/P2 学术论证）                → 形成"快·准·知不准·风险可解释"
```

**建议顺序**：先 A（几天出结果，能交差），同时把 B 当申报书"现状"素材，C 是你大创真正的创新落点。

---

## 八、你需要做的事（沙箱做不了：本机 + 账号）

1. **复现 U-FNO（首选）**：
   - `git clone https://github.com/gegewen/ufno` 到 `C:\workplace\ccus-ai-surrogate\repro\ufno`
   - 用 Google 账号下数据集（README 里的 Drive 链接），放进仓库根目录
   - 用你已装好的 venv 跑 `train_ufno_gas_saturation.py`（⚠️ 本机执行，沙箱无 GUI/Google 账号）
   - 复现靶：gas saturation 相对误差低个位数 %、R²>0.99
2. **读两篇关键论文**（本机访问）：
   - U-FNO：Advances in Water Resources 163 (2022) 104180
   - Zingaro 2026：arXiv:2606.07215（下载 PDF 读 §3 实验设置 + §4 架构对比）
3. **决定复现深度**：只做 A（代码复现，快）还是进 B（方法学 benchmark，改数据管线）。建议先 A 后 B。
4. **（阶段 B 才需要）** 改造 `sim/synthetic_field.py` 生成异质渗透率场——这一步我会和你对齐后再动手。

> 注：凡涉及 git clone / Google Drive 下载 / 跑 GUI notebook，都需你本机操作（沙箱无账号、无 GUI）。复现出的指标落到 `C:\workplace\ccus-ai-surrogate\repro\ufno\` 下，我们再对接你的 UQ/风险模块做优化。
