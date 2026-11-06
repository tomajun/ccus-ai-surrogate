# -*- coding: utf-8 -*-
"""
生成《CO2 地质封存智能预警项目 — 组内介绍》Word 文档。
步骤：
  1) 用 matplotlib 画"六种泄漏机制"剖面示意图 -> docs/泄漏机制示意.png
  2) 用 python-docx 组装文档 -> docs/项目组内介绍_CO2封存智能预警.docx
运行：
  .venv/Scripts/python.exe tools/make_intro_doc.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)
PNG = os.path.join(DOCS, "泄漏机制示意.png")
DOCX = os.path.join(DOCS, "项目组内介绍_CO2封存智能预警.docx")

# ----------------------------------------------------------------------------
# 1) 示意图
# ----------------------------------------------------------------------------
def set_cjk_font():
    # Windows 中文环境通常有 SimHei / Microsoft YaHei
    for cand in ["SimHei", "Microsoft YaHei", "Microsoft JhengHei", "Arial Unicode MS"]:
        try:
            font_manager.findfont(cand, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [cand]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

def draw_diagram():
    set_cjk_font()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 地层
    layers = [
        (0.0, 0.8, "#5b5b5b", "基底"),
        (0.8, 3.0, "#e6c79c", "储层（CO2 封存层）"),
        (3.8, 1.0, "#8a8a8a", "盖层（封闭层）"),
        (4.8, 1.5, "#bcdce8", "浅层含水层"),
        (6.3, 0.7, "#eef7fb", "地表 / 大气"),
    ]
    for y0, h, color, label in layers:
        ax.add_patch(patches.Rectangle((0, y0), 10, h, facecolor=color, edgecolor="#333", lw=0.6))
        ax.text(0.15, y0 + h / 2, label, fontsize=10, va="center",
                color="#222", fontweight="bold")

    # 地表线
    ax.plot([0, 10], [6.3, 6.3], color="#2b6c8f", lw=1.2, ls="--")

    # 注入井
    ax.plot([5, 5], [7.0, 1.0], color="#111", lw=2.4)
    ax.text(5.18, 6.72, "注入井", fontsize=9, ha="left", color="#111", fontweight="bold")

    # 废弃井筒（穿透盖层）
    ax.plot([2.2, 2.2], [6.3, 1.0], color="#444", lw=2.0, ls=":")
    ax.text(2.45, 6.42, "废弃井筒", fontsize=9, ha="left", color="#444")

    # 断层（穿越盖层）
    ax.plot([7.4, 8.6], [0.8, 4.8], color="#9b6b4a", lw=2.0)
    ax.text(8.68, 5.1, "断层", fontsize=9, color="#9b6b4a", fontweight="bold")

    # 侧向边界
    ax.plot([9.4, 9.4], [0.8, 3.8], color="#b00020", lw=1.4, ls="--")
    ax.text(9.0, 3.95, "封存体边界", fontsize=8.5, color="#b00020", ha="right")

    # CO2 羽流
    plume = patches.Ellipse((5, 2.3), 2.6, 2.6, facecolor="#e23b3b", alpha=0.30,
                            edgecolor="#c0202f", lw=1.2)
    ax.add_patch(plume)
    ax.annotate("", xy=(5, 3.0), xytext=(5, 1.6),
                arrowprops=dict(arrowstyle="->", color="#c0202f", lw=1.6))
    ax.text(5, 2.3, "CO2\n羽流", fontsize=9, ha="center", va="center",
            color="#7a0f17", fontweight="bold")

    # 泄漏路径箭头（红色）
    arrows = [
        ((5.0, 3.7), (5.0, 4.7)),     # 1 盖层
        ((2.2, 2.0), (2.2, 6.2)),     # 2 废弃井上行
        ((7.9, 2.6), (8.2, 4.6)),     # 3 断层
        ((6.2, 2.3), (9.2, 2.3)),     # 4 侧向
        ((3.8, 3.9), (3.8, 5.2)),     # 5 溶解入侵含水层（穿入含水层带）
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#b00020", lw=1.8))

    # 地表渗漏（⑥）：废弃井顶部小喷发
    ax.annotate("", xy=(2.2, 6.95), xytext=(2.2, 6.55),
                arrowprops=dict(arrowstyle="->", color="#b00020", lw=1.8))

    # 编号标记
    markers = [
        (5.0, 4.55, "1"),
        (2.2, 3.0, "2"),
        (8.05, 3.6, "3"),
        (9.4, 2.3, "4"),
        (3.8, 5.35, "5"),
        (2.2, 6.75, "6"),
    ]
    for x, y, n in markers:
        ax.add_patch(patches.Circle((x, y), 0.18, facecolor="#b00020", edgecolor="white", lw=1.2))
        ax.text(x, y, n, fontsize=9, ha="center", va="center", color="white", fontweight="bold")

    ax.set_title("CO2 地质封存：六种泄漏机制剖面示意（编号 ①~⑥ 对应正文表格）",
                 fontsize=12, fontweight="bold", color="#222", pad=10)
    fig.tight_layout()
    fig.savefig(PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("diagram ->", PNG)

# ----------------------------------------------------------------------------
# 2) Word 文档
# ----------------------------------------------------------------------------
def add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(9.5)
    return t

def build_doc():
    doc = Document()
    # 标题
    title = doc.add_heading("CO₂ 地质封存智能预警项目", level=0)
    sub = doc.add_paragraph()
    r = sub.add_run("—— 组内介绍（项目主题 · 六种泄漏机制 · U-FNO 复现 · 三级预警模型）")
    r.italic = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    meta = doc.add_paragraph()
    mr = meta.add_run("大创项目 · 方法定位：复现 SOTA（U-FNO）＋ 不确定性感知三级预警 · 内部资料")
    mr.font.size = Pt(9)
    mr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # 一、项目主题
    doc.add_heading("一、项目主题", level=1)
    doc.add_paragraph(
        "我们要做一件什么事？用深度学习搭一个“快得多的 AI 替代模拟器”：给定储层的岩石属性"
        "（渗透率、孔隙度场）和注入条件（注入速率、时间），它能在毫秒级预测 CO₂ 注入地下后的"
        "羽流（气相饱和度）如何随时间和空间运移，并进一步给出“泄漏风险”的三级预警。")
    doc.add_paragraph(
        "为什么值得做？CCUS（碳捕集、利用与封存）是实现“双碳”目标的关键技术，而地质封存的核心"
        "安全问题就是：注入的 CO₂ 会不会漏出来。传统的数值模拟（如 TOUGH、ECLIPSE）跑一次要"
        "几小时到几天，无法支撑大量情景筛查与实时决策；神经网络代理模型（surrogate）可以把这件事"
        "加速成千上万倍。")
    doc.add_paragraph(
        "项目定位（诚实版）：这是一个本科生大创项目。我们不宣称方法首创，而是以“复现前沿 SOTA "
        "模型（U-FNO）+ 在其上叠加深度集成不确定性量化（UQ）”作为可辩护的贡献，并把代理模型与"
        "风险筛查串成一个完整工作流。")

    # 二、六种泄漏机制
    doc.add_heading("二、流程示意：CO₂ 封存的六种泄漏机制", level=1)
    doc.add_paragraph(
        "下图是一个简化的地质封存剖面。CO₂ 从注入井进入深部储层后，若圈闭失效，可能经由六类机制"
        "泄漏到上方或地表（编号 ①~⑥，对应下方表格）。这六种机制是 NRAP（美国碳封存风险框架）的"
        "标准分类，也是我们“预警模型”要识别的对象。")
    doc.add_picture(PNG, width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cr = cap.add_run("图 1：CO₂ 地质封存六种泄漏机制剖面示意（红圈为泄漏通道位置，红箭头为泄漏方向）")
    cr.italic = True
    cr.font.size = Pt(9)
    cr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_table(doc,
              ["编号", "泄漏机制", "触发条件 / 泄漏路径", "模型如何感知"],
              [
                  ["①", "盖层密封失效", "注入压力过高或盖层局部薄弱，CO₂ 突破顶部封闭盖层",
                   "预测羽流触及 / 穿过盖层带"],
                  ["②", "废弃井筒泄漏", "历史遗留或废弃井穿透盖层，成为 CO₂ 上窜通道",
                   "羽流与已知井筒位置重叠并沿其上行"],
                  ["③", "断层活化", "注入压力触发断层滑动，打开原本封闭的通道",
                   "羽流靠近 / 沿断层带迁移"],
                  ["④", "侧向越界运移", "羽流横向迁移到封存体边界之外",
                   "预测羽流抵达模型边界"],
                  ["⑤", "溶解 / 卤水入侵含水层", "CO₂ 溶于卤水，随地下水污染浅层含水层",
                   "上层含水层出现 CO₂ 信号"],
                  ["⑥", "近地表 / 地表渗漏", "上述任一通道贯通至地表，造成地表释放",
                   "泄漏路径贯通至地表"],
              ])

    # 三、U-FNO 复现
    doc.add_heading("三、我们要复现的内容：U-FNO", level=1)
    doc.add_paragraph(
        "U-FNO 是什么？它是 Wen 等人（2022，Advances in Water Resources）提出的“上采样 Fourier "
        "Neural Operator（傅里叶神经算子）”，是地下流动代理模型的当前 SOTA 方法。它的输入是储层"
        "静态属性（渗透率 / 孔隙度等场）＋ 坐标 / 时间编码，输出是 CO₂ 气相饱和度（即羽流）的"
        "时空演化。简单说：它把“给一张地质图，算出 CO₂ 云怎么飘”这件事，从几小时压到毫秒。")
    doc.add_paragraph(
        "复现目标：在作者公开的数据集（CO₂ 两相流，2D 剖面约 200×96 网格 × 24 个时间帧）上，把"
        "相对 L2 误差、决定系数 R² 复现到论文水平（R² > 0.99）。")
    doc.add_paragraph(
        "为什么要“复现”而不是直接开做？因为这是整座项目的地基。我们此前在自合成玩具数据上验证过："
        "模型会把羽流学成“全零”，既学不动点预测、也学不出不确定性——那是一条死路。只有在真实、"
        "有信号的数据上把 U-FNO 跑通，后续工作才有意义。")
    doc.add_paragraph(
        "复现产物：训练 / 评估脚本（已把官方 Jupyter notebook 转写为可直接运行的 .py，CPU / GPU 均可）、"
        "复现指标（相对 L2 误差、R²）。")

    # 四、三级预警模型
    doc.add_heading("四、复现基础上的工作：三级预警（不确定性感知）模型", level=1)
    doc.add_paragraph(
        "U-FNO 只给出一个“点预测”——一张最可能的羽流场。但工程决策真正需要的是“这个结果可不可信”。"
        "深度学习在分布外（out-of-distribution）场景下往往会过度自信。因此我们要在 U-FNO 之上叠加"
        "不确定性量化（UQ）。")
    doc.add_paragraph(
        "做法（深度集成 UQ）：训练 5 个 U-FNO 成员（用 bootstrap 重采样让每个成员看到略不同的数据），"
        "用这 5 个成员的预测方差作为不确定性。校准良好的标志是：真实值落在预测区间内的比例（PICP）"
        "约 90%。这一步是 U-FNO 原论文没有做的，是我们的核心增量，且有 2025 年 SPE 论文"
        "（Pawar 的 U-FNO＋蒙特卡洛 UQ 场址筛选、Velasco Lozano 的 FNO＋泄漏风险 UQ）作为文献背书。")
    doc.add_paragraph(
        "三级预警规则（透明、可解释的启发式，由“点预测 + 不确定性带”导出）：")
    add_table(doc,
              ["等级", "触发条件", "含义与建议动作"],
              [
                  ["绿（低风险）", "预测羽流被稳妥圈闭在储层内，且不确定性带不触及盖层 / 断层 / 边界 / 井筒",
                   "圈闭良好，可按计划注入"],
                  ["黄（中风险）", "预测羽流接近盖层 / 断层 / 边界，或不确定性带触及上述结构",
                   "需加密监测、降低注入速率或复核地质模型"],
                  ["红（高风险）", "预测羽流直接触及 / 突破盖层、断层、废弃井或边界，或不确定性带覆盖关键泄漏通道",
                   "暂停 / 调整注入，启动泄漏应急响应预案"],
              ])
    note = doc.add_paragraph()
    nr = note.add_run(
        "诚实声明：本预警是基于“点预测 + 不确定性”的启发式筛查工具，用于辅助决策，并非认证级风险"
        "评估模型；其风险分级规则由项目组定义，缺乏真实泄漏事件作为真值校验，应在报告中明确标注这一局限。")
    nr.font.size = Pt(9)
    nr.font.color.rgb = RGBColor(0x99, 0x33, 0x33)

    doc.add_paragraph()
    foot = doc.add_paragraph()
    fr = foot.add_run(
        "下一步（组内待办）：① 本机下载 U-FNO 公开数据集并在 Colab（免费 GPU）跑通复现；"
        "② 问导师“复现 + UQ 增量”是否达到大创结题要求；③ 在真实数据上叠加深度集成 UQ 并标定三级阈值。")
    fr.font.size = Pt(9)
    fr.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.save(DOCX)
    print("docx  ->", DOCX)

if __name__ == "__main__":
    draw_diagram()
    build_doc()
