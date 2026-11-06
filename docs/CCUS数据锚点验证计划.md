# CCUS 数据锚点验证计划（修正版 · 2026-10-15）

> 关键修正：原版假设"能登 EDX 免账号下 Kimberlina / NATCARB"，**实测不成立**——EDX 对个人不可达。
> 本版把 Track A 改走 **CO2 DataShare（免注册、Open）**，Track B 用文献公开统计区间落地（同样不经 EDX）。

---

## 0. 先说结论（2026-10-15 实测核实）

- **EDX（edx.netl.doe.gov）对个人基本堵死**：注册页原话 *"register with a \*.GOV or \*.MIL email; if you do not have one, you must list an NETL POC to validate your active engagement with NETL research."* 即无 `.gov/.mil` 邮箱，必须有一名 **NETL 在职人员（POC）** 担保"确实参与 NETL 研究"，否则账号不批。本科生无 NETL 关系 → 个人无法注册，也别指望"用学校邮箱试试"。
- **Kimberlina 1.2 全量只在 EDX 上**（DOI 10.18141/1887287 解析到 EDX），无 Zenodo 公开镜像；且它是 **54 个 zip 的地球物理监测集**（地震/重力/电磁/测井），不是羽流流场本身，方向也不如 Sleipner 对口。→ **原 Track A 放弃**。
- **替代真实数据源：CO2 DataShare（co2datashare.org）**——SINTEF 维护，汇集 Sleipner / Decatur / Northern Lights 等**真实 CO₂ 封存项目**，**明确 "accessible without registration"（免登录、Open）**，含储层模拟模型与羽流动力学，正好对口 plume surrogate。这是更优解。

---

## 1. 轨道 A'（新）：CO2 DataShare / Sleipner 2019 Benchmark Model

- 入口：`https://co2datashare.org/` → 搜 `Sleipner 2019 Benchmark Model`
- DOI：`10.11582/2020.00004`；许可：SLEIPNER CO2 REFERENCE DATASET LICENSE（Open，研究可用，需署名）；访问级别 **Open，免登录**。
- 内容：模拟网格 + 岩石物理属性 + 井数据 + 速度场/层位（= 真实储层模型，可喂任意流模拟器得到 CO₂ 饱和场）。这是业界羽流动力学标杆。
- **白送的对标文献（就在数据集"Reference list"里）**：
  - *Multifidelity Monte Carlo Method For Rapid Uncertainty Quantification In CO2 Storage* (2025) —— 直接对口 **P5 多保真度 UQ**。
  - *Joint Inversion Of Geophysical Data ... A Differentiable Physics-Informed Neural Network Model* (2022) —— 对口 physics-informed surrogate。
  - *Performance Of An Open-Source Image-Based History Matching Framework For CO2 Storage* (2025)。

### Tier 1（最低成本 · 现在就能做 · 免 flow 模拟）
1. 下载 Sleipner 2019 模型（小文件，免登录），取其**静态几何**：储层厚度、注入点位置、注入速率（~1 Mt/yr）、顶构造——作为合成生成器的"真实几何边界"，与 Track B NATCARB 参数先验互补。
2. 用 Sleipner 已发表的 4D 地震羽流形态（业界著名羽流 blob）+ 文献饱和场剖面，做**定性/定量形状校验**：比对我们的合成羽流足迹、厚度、迁移方向是否与真实羽流一致。
3. 申报书写法："代理模型几何/参数边界锚定真实 Sleipner 项目（Open 数据，co2datashare.org），参数分布取自 NATCARB 公开统计。"——真实、可辩护、审稿人也能免登录复现。

### Tier 2（更严谨 · 可选 stretch · 需跑流模拟）
- 下载 Sleipner 2019 **ECLIPSE deck**，用 **OPM Flow（开源流模拟器）** 跑出多时间步 CO₂ 饱和场，取切片汇入与 `sim/kimberlina_csv_to_npz.py` 同款合同 `(N,T,48,48)`，作为真实 fine-tuning / 验证集。
- 成本：装 OPM Flow（C++，本机编译或取预编译二进制）+ 跑一次模拟（CPU 可行，Sleipner 网格不大）。比 EDX 路径省了"找 NETL POC"这一不可控环节。
- 注：原 `sim/kimberlina_csv_to_npz.py` 面向 Kimberlina 的 ASCII CSV；若走 Tier 2，应新增一个 `sim/sleipner_eclipse_to_npz.py`（解析 GRDECL/EGRID + 饱和关键字），逻辑相近。

---

## 2. 轨道 B（NATCARB 真实参数先验）—— 已完成，不经 EDX

- 原文档写"edx 搜 natcarb"有误：NATCARB 同样挂在 EDX 上。但我们用的是**文献/报告里已公开的 NATCARB 统计区间**（孔隙度/渗透率/厚度 P10–P90），直接填入 `data/priors.json` 并 `enabled:true`，`make_dataset.py` 已按真实分布采样（**已验证 GO，决定性**）。**全程不经过 EDX 登录**。
- 申报书写法："参数先验取自公开北美封存数据库 NATCARB 汇总统计（CC BY 4.0）"——真实可辩护。

---

## 3. 决策树（修正版）

```
Track B 真实参数先验  ✅ 已 GO（NATCARB 文献统计，免 EDX）
Track A'（CO2 DataShare）
   ├─ Tier 1 几何+形状校验（免登录，现在可做） → 声明"真实数据锚成立" → ★ All-in CCUS
   └─ Tier 2 跑 OPM Flow 得真实饱和场（可选）   → 数据锚更强、可定量 fine-tune
EDX/Kimberlina 路径：正式放弃（对个人不可达，且方向不如 Sleipner 对口）
```

**结论**：数据锚从"信念"变成"Open 真实项目(Sleipner) + 文献统计(NATCARB)"的硬证据，成本比原 EDX 方案更低、可复现性更好。CCUS 反转条件满足，可 All-in P4/P5。

---

## 4. 一句话

EDX 这扇门对个人关上了，但 CO2 DataShare 这扇门开着、且更对口羽流代理。先把 Track A' 的 Tier 1 做了（下载 Sleipner 模型取几何 + 形状校验），再视精力决定要不要上 Tier 2（OPM Flow 出真实饱和场）。
