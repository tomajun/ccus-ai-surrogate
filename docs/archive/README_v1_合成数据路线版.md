# CCUS-AI-Surrogate · CO₂ 封存羽流快速预测与泄漏风险智能评估

> 国家级/省级大学生创新创业训练计划（创新类）开源作品
> 方向：**CCUS（碳捕集、利用与封存）** × **AI 代理建模**
> 一句话定位：用开源 AI 模型，把 CO₂ 地质封存的羽流演化从"小时级数值仿真"压缩到"秒级预测"，并给出泄漏风险智能评级。

---

## 1. 背景与意义
在"双碳"战略下，CCUS 是把工业 CO₂ 捕下并封入地下数千米岩层的关键技术。传统做法靠数值模拟（如 MRST/PorePy）预测 CO₂ 羽流怎么迁移、会不会泄漏，单次仿真动辄数小时~数天，难以支撑选址比选与实时安全评估。**本项目用深度学习代理模型（Surrogate Model）学习"注入参数 → 羽流时空演化"的映射，实现毫秒~秒级预测，并叠加泄漏风险智能分级**，为 CCUS 选址与安全性论证提供轻量、可解释、开源的工具原型。

## 2. 核心创新点
| 层级 | 内容 | 相对 GitHub 同类 |
|---|---|---|
| 基础版 | 合成数据 + CNN 代理模型预测 CO₂ 场 | CCUS 开源 ML 极少，本方向近乎蓝海 |
| **进阶 A** | **傅里叶神经算子（FNO）代理建模**，泛化更优 | 仅 CO2-hysteresis-neural-operator(1★) 刚起步 |
| **进阶 B** | **泄漏风险智能分级**（MLP 三档：低/中/高） | CO2_Leakage_ML-LCA(2★) 极少 |
| 进阶 C | 交互式 Web 演示（Streamlit）+ 可解释 | 几乎空白 |

> 调研依据：检索 `CCUS carbon capture` / `CO2 geological storage` / `carbon capture machine learning` 的 GitHub 全库匹配数分别为 **24 / 29 / 25**，且绝大多数为 0–8★ 的 MATLAB 仿真或材料筛选仓库，"AI 代理 + 风险评估"细分近乎无人占据。

## 3. 技术路线
```
数据层   内置教学级合成模拟器(sim/synthetic_field.py) 生成 (注入参数→CO₂场序列)
         （高保真可替换为 MRST/MATLAB 或 PorePy/Python）
模型层   基线: UNetLite(CNN)  |  进阶: FNO(傅里叶神经算子)
评估层   RMSE / SSIM / 推理延迟 / 概念加速比
风险层   RiskMLP: 参数+场统计 → 低/中/高 三档
交付层   Streamlit 交互演示（拖动滑块看羽流+风险）
```

## 4. 数据集与获取
- **本仓库不含任何真实涉密数据。** 全部训练样本由 `sim/make_dataset.py` 用内置合成模拟器生成（物理合理、零保密风险）。
- 论文中可注明：合成数据为"教学级代理数据"；如需高保真，按 README 第 7 节替换为 MRST/PorePy 导出即可。
- 数据许可：**CC0**（生成数据可自由使用）。

## 5. 快速开始

> **提速关键**：默认 `requirements.txt` 里的 `torch` 会从 PyPI 拉**带 CUDA 的完整版（~800MB+）**，这是慢的主因。请用下面的 **CPU 专用源** 安装（~190MB），笔记本训练完全够用。仓库已内置 `.venv` 配置示例见 `tools/setup_env.bat`（Windows）/ `tools/setup_env.sh`（Linux/Mac）。

**推荐做法（Windows，约 2–4 分钟）：**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 先装 CPU 版 torch(约190MB)
pip install numpy matplotlib scikit-learn pytest streamlit -i https://mirrors.aliyun.com/pypi/simple  # 其余走阿里镜像提速
pytest tests/                                                  # 测试

# —— 当前交付路线：官方 U-FNO 权重 + eval 推理（不训练）——
# 数据集需自行按 docs/数据集与结题可行性审计.md 下载(约 20 GB)放到 数据集/
python tools/run_uq_risk.py --model_path 数据集/sg_UFNO_4500_trained.pt --data_dir 数据集 \
       --limit 500 --replicas 8 --alpha 0.10 \
       --out_json experiments/ufno_uq_report_500.json --out_csv experiments/ufno_uq_tiers_500.csv   # 推理→扰动→保形→判级+可靠性扫描，约 41 分钟
python tools/make_figures.py                                   # 读上面的产物出 9 张数据图 → results/ufno_uq/
streamlit run src/app/streamlit_app.py                         # 演示(只读 json/csv/npz 产物，不需要 torch/GPU)

# —— 早期合成数据路线（已停用，仅保留可复现的历史脚手架）——
# python sim/make_dataset.py --n 800 --out data/synthetic.npz
# python -m src.train --model cnn --epochs 20
```

**一次性脚本（等价上面的步骤）：** `tools/setup_env.bat`（Windows）。也可直接 `pip install -r requirements.txt`，但会装 CUDA 版 torch，下载更慢。

## 6. 结果指标（正式数字以 `experiments/ufno_uq_report_500.json` 为准）
| 项目 | 数值 | 说明 |
|---|---|---|
| U-FNO 复现精度 | rel L2 = **0.0647** | 官方权重 + eval  split 500 样本，与官方仓库口径一致 |
| 单场地端到端 | ≈ **5.0 s**（500 场地 41 分钟） | 推理 + 8 次参数扰动 + 保形校准 + 判级 + 8 档可靠性扫描（RTX 3070 Laptop） |
| 90% 区间覆盖率 | PICP = **0.911** | 羽流内格点、split conformal 交付口径 |
| 三级判级 | 绿 **375** / 黄 **75** / 红 **50** | 500 场地，真值口径（图 3 展示区间口径下的"确定档"占比） |

> 早期合成数据路线的 RMSE/SSIM 表已停用，历史产物仍在 `experiments/metrics_*.json`。

## 7. 高保真数据替换（进阶）
本仓库用 Python 合成数据保证"开箱即跑"。若需贴近真实物理，可：
- **MRST（MATLAB, 学术免费）**：`lsalo/mrst-adblackoil-gcs`、`salmansust/CO2-Sequestration` 提供 CO₂ 封存仿真示例，导出 `.mat` 后用本仓库已实现的 `sim/export_to_numpy.py` 一键转为统一训练样本（`pip install -r requirements-hifi.txt` 提供 scipy）。
- **PorePy（Python, SINTEF 开源）**：纯 Python 裂缝性储层仿真，可直接产出 `.npy/.npz`，同样用 `sim/export_to_numpy.py` 转换。
- 转换后 `train.py` / `train_risk.py` / `dataset.py` **零改动**即可吃高保真数据。字段约定与 `sim/make_dataset.py` 完全一致：`fields/params/risks`。详见 `sim/export_to_numpy.py` 文件头注释（含 `params.json` 写法）。

## 8. 仓库结构
```
ccus-ai-surrogate/
├── README.md / LICENSE(MIT) / CONTRIBUTING.md / requirements.txt / .gitignore
├── sim/           合成数据生成器 + 数据集脚本
├── src/
│   ├── dataset.py / dataset_b.py     数据集封装
│   ├── models/    cnn_surrogate / fno_surrogate / risk_classifier
│   ├── train.py / train_b.py         训练 + 评估
│   ├── train_uq.py                   深度集成 UQ（合成数据路线，已停用）
│   ├── uq_model.py                   UQ 引擎：参数扰动 MC + split conformal 校准
│   ├── risk_model.py                 三级预警：绝对几何带(旧) + 相对分位/区间判级(新口径)
│   ├── risk_model_demo.py            命令行演示：timeline / calibrate / tier
│   └── app/       streamlit_app.py — 5 页演示，只读 experiments/ 产物 + results/ 图
├── repro/ufno/    官方 U-FNO 复现（eval + 按 stride 分页直读 ufno_reader.py）
├── tools/         一键管线 run_uq_risk.py（推理→扰动→保形→判级）、make_figures.py（9 张数据图）、bench_gpu_inference.py
├── data/          仅放生成脚本输出(npz 被 gitignore)
├── 数据集/         官方 U-FNO 权重与 eval 样本(~20 GB，需按审计文档下载)
├── experiments/   实验配置与指标 json / 判级 csv / 逐场地 npz
├── results/ufno_uq/ fig1~fig9 数据图(散点对比、残差、区间跨度、分档对比、场带、时间轴、可靠性、羽流演化剖面、判级混淆矩阵)
├── docs/          讲解文档 / 实验方案 / 答辩材料
└── tests/         单元测试（test_synthetic.py / test_uq_risk.py）
```

## 9. 贡献与协议
- 提交规范：Conventional Commits（`feat:`/`fix:`/`docs:`/`data:`/`exp:`/`refactor:`/`test:`）
- 分支：`main`(稳定) + `dev`(集成) + `feature/*`
- 协议：代码 **MIT**；数据/文档 **CC0 / CC-BY-4.0**
- 参考项目见 `docs/参考项目清单.md`（同申报书方案文档）

## 10. 团队与致谢
（此处填写团队成员、指导教师、所属院校）
致谢开源社区：MRST、PorePy、neuralop、scikit-learn、streamlit。

---
*本仓库所有代码、数据生成脚本、实验图表、申报文档均开源托管，符合大创"公开、可复现"要求。*
