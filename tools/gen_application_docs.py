# -*- coding: utf-8 -*-
"""
生成两份 Word 文档：
1) 大创申请书（基于 CCUS 项目改写模板；个人信息全部用【】占位，不泄露任何标识）
2) 文献真实性查验清单（含检索关键词 + 入口，供学校文献库复核）

运行：
  .venv\Scripts\python.exe tools\gen_application_docs.py
"""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUT_APP = os.path.join(DOCS, "大创申请书_面向CO2地质封存代理模型与风险预警.docx")
OUT_LIT = os.path.join(DOCS, "文献真实性查验清单.docx")
DIAG = os.path.join(DOCS, "泄漏机制示意.png")


def set_cjk(run, font="宋体"):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font)
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)


def style_base(doc):
    st = doc.styles["Normal"]
    st.font.size = Pt(10.5)
    st.font.name = "宋体"
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "宋体")
    rfonts.set(qn("w:ascii"), "宋体")
    rfonts.set(qn("w:hAnsi"), "宋体")


def add_heading4(doc, text):
    p = doc.add_paragraph()
    p.style = doc.styles["Heading 4"]
    r = p.add_run(text)
    set_cjk(r, "黑体")
    r.font.size = Pt(12)
    r.bold = True
    return p


def add_subtitle(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_cjk(r, "黑体")
    r.bold = True
    r.font.size = Pt(11)
    return p


def add_para(doc, text, size=10.5, bold=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    r = p.add_run(text)
    set_cjk(r)
    r.font.size = Pt(size)
    r.bold = bold
    return p


def set_cell(cell, text, bold=False, size=10, align_center=False):
    cell.text = ""
    p = cell.paragraphs[0]
    if align_center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_cjk(r)
    r.font.size = Pt(size)
    r.bold = bold


def merge_row_center(table, row_idx, text, ncols, size=10):
    """将第 row_idx 行从 col1 起合并到最后一列，写入 text。"""
    cell = table.cell(row_idx, 1)
    for c in range(2, ncols):
        cell = cell.merge(table.cell(row_idx, c))
    set_cell(cell, text, bold=False, size=size)


# ======================================================================
# 文档一：大创申请书（CCUS 项目，个人信息占位）
# ======================================================================
def build_application():
    doc = Document()
    style_base(doc)

    # 标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rt = title.add_run("大学生创新训练项目计划申请书")
    set_cjk(rt, "黑体")
    rt.bold = True
    rt.font.size = Pt(16)

    # 填写说明（通用文本，无个人信息）
    doc.add_paragraph()
    add_para(doc, "填写说明", bold=True)
    add_para(doc, "本申请书所列各项内容均须实事求是，认真填写，表达明确严谨，简明扼要。")
    add_para(doc, "申请人可以是个人，也可为创新团队，首页只填负责人。“项目编号”一栏不填。")
    add_para(doc, "本申请书为大16开本（A4），左侧装订成册。可网上下载、自行复印或加页，"
                  "但格式、内容、大小均须与原件一致。")
    add_para(doc, "负责人所在学院认真审核, 经初评和答辩，签署意见后，将申请书（一式两份）报送项目管理办公室。")

    # 一、基本情况
    add_heading4(doc, "一、基本情况")
    t0 = doc.add_table(rows=8, cols=4)
    t0.style = "Table Grid"
    t0.alignment = WD_TABLE_ALIGNMENT.CENTER

    def set_label(cell, text):
        set_cell(cell, text, bold=True, size=10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    rows0 = [
        ("项目编号", "（不填）"),
        ("项目名称", "面向CO₂地质封存的轻量级深度学习代理模型与泄漏风险不确定性量化"),
        ("项目负责人", "【负责人姓名】"),
        ("所在学院", "【所在学院】"),
        ("学号", "【学号】"),
        ("指导教师", "【指导教师姓名】"),
        ("E-mail", "【指导教师邮箱】"),
        ("申请日期", "【申请日期】"),
    ]
    # 行布局：标签在 col0；col1 起为值（多列合并）
    for i, (lab, val) in enumerate(rows0):
        set_label(t0.cell(i, 0), lab)
        if lab in ("项目名称", "所在学院", "指导教师", "E-mail"):
            merge_row_center(t0, i, val, 4)
        elif lab == "项目编号":
            merge_row_center(t0, i, val, 4)
        elif lab == "项目期限":
            set_cell(t0.cell(i, 1), val)
            set_cell(t0.cell(i, 2), "项目来源", bold=True)
            t0.cell(i, 2).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell(t0.cell(i, 3), "■ A、学生自主选题  □ B、教师科研项目  □ C、企业委托  □ D、拔尖专项  □ E、竞赛专项  □ F、研修专项")
        elif lab == "申请日期":
            set_cell(t0.cell(i, 1), val)
            set_cell(t0.cell(i, 2), "项目期限", bold=True)
            t0.cell(i, 2).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell(t0.cell(i, 3), "二年期")
        elif lab == "项目负责人":
            set_cell(t0.cell(i, 1), val)
            set_cell(t0.cell(i, 2), "联系电话", bold=True)
            t0.cell(i, 2).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell(t0.cell(i, 3), "【联系电话】")
        elif lab == "学号":
            set_cell(t0.cell(i, 1), val)
            set_cell(t0.cell(i, 2), "专业班级", bold=True)
            t0.cell(i, 2).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell(t0.cell(i, 3), "【专业班级】")

    # 二、立项依据
    add_heading4(doc, "二、立项依据")

    # 详细基本情况表（原模板 19x9 信息表的精简忠实复刻）
    t1 = doc.add_table(rows=12, cols=4)
    t1.style = "Table Grid"
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER

    def lbl(cell, t):
        set_cell(cell, t, bold=True, size=9.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 0 项目名称
    lbl(t1.cell(0, 0), "项目名称")
    merge_row_center(t1, 0, "面向CO₂地质封存的轻量级深度学习代理模型与泄漏风险不确定性量化", 4, size=9.5)
    # 1 所属学科
    lbl(t1.cell(1, 0), "所属学科")
    merge_row_center(t1, 1, "学科一级类：工学    学科二级类：计算机类", 4, size=9.5)
    # 2 申请金额 / 项目期限
    lbl(t1.cell(2, 0), "申请金额")
    set_cell(t1.cell(2, 1), "3000元")
    lbl(t1.cell(2, 2), "项目期限")
    set_cell(t1.cell(2, 3), "二年期")
    # 3 负责人
    lbl(t1.cell(3, 0), "负责人")
    set_cell(t1.cell(3, 1), "【姓名】")
    lbl(t1.cell(3, 2), "性别")
    set_cell(t1.cell(3, 3), "【】")
    # 4 学号 / 联系电话
    lbl(t1.cell(4, 0), "学号")
    set_cell(t1.cell(4, 1), "【学号】")
    lbl(t1.cell(4, 2), "联系电话")
    set_cell(t1.cell(4, 3), "宅：【】  手机：【】")
    # 5 指导教师 / 联系电话
    lbl(t1.cell(5, 0), "指导教师")
    set_cell(t1.cell(5, 1), "【姓名】")
    lbl(t1.cell(5, 2), "联系电话")
    set_cell(t1.cell(5, 3), "宅：【】  手机：【】")
    # 6 项目简介（整行）
    lbl(t1.cell(6, 0), "项目简介")
    intro = ("随着“双碳”目标的推进，二氧化碳地质封存（CCS/CCUS）成为大规模减排的关键技术路径。"
             "封存安全性评估高度依赖对CO₂羽流运移的高保真数值模拟，但多相流数值模拟单次运行耗时长、计算成本高，"
             "难以支撑场地筛选、不确定性量化与实时风险预警等需要海量正演模拟的工作。近年来，以傅里叶神经算子（FNO）"
             "为代表的算子学习方法，能够从数值模拟数据中学习“渗透率/孔隙度/注入参数 → CO₂饱和度时空演化”的映射，"
             "在保证精度的同时实现千倍级加速。本项目拟以U-FNO（增强型傅里叶神经算子）为基线，复现其在CO₂两相流"
             "气相饱和度预测上的性能；在此基础上引入深度集成（Deep Ensemble）不确定性量化，为羽流预测提供校准的区间估计；"
             "进一步结合封存场地地质几何（盖层、废弃井筒、断层、封存体边界），构建一套轻量级、可解释的泄漏风险"
             "三级（绿/黄/红）预警模型，将“快速代理—不确定性—风险预警”串联为可演示的工作流。")
    merge_row_center(t1, 6, intro, 4, size=9.5)
    # 7 负责人曾经参与科研的情况
    lbl(t1.cell(7, 0), "负责人曾经参与科研的情况")
    merge_row_center(t1, 7, "【由负责人据实填写】", 4, size=9.5)
    # 8 指导教师承担科研课题情况
    lbl(t1.cell(8, 0), "指导教师承担科研课题情况")
    merge_row_center(t1, 8, "【由指导教师据实填写】", 4, size=9.5)
    # 9 指导教师对本项目的支持情况
    lbl(t1.cell(9, 0), "指导教师对本项目的支持情况")
    merge_row_center(t1, 9, "【由指导教师据实填写：含算力/实验环境、关键技术指导、项目管理等方面】", 4, size=9.5)
    # 10 项目组主要成员（表头）
    lbl(t1.cell(10, 0), "项目组主要成员")
    for j, h in enumerate(["姓名", "学号", "学院", "专业班级", "联系电话", "项目分工"]):
        set_cell(t1.cell(10, j), h, bold=True, size=9.5, align_center=True)
    # 11 成员占位行（合并前6列写提示）
    merge_row_center(t1, 11, "【姓名】【学号】【学院】【专业班级】【联系电话】【分工：模型复现 / UQ / 风险模型 / 数据集 / 集成演示】（成员数量由实际组队决定）", 4, size=9.5)

    doc.add_paragraph()

    # 立项依据正文
    add_subtitle(doc, "（一）研究目的")
    add_para(doc, "二氧化碳地质封存是实现“双碳”目标的关键减排技术，其长期安全性取决于对注入CO₂在储层中"
                  "运移、圈闭与潜在泄漏路径的准确评估。传统高保真多相流数值模拟（如TOUGH、ECLIPSE、GEOS等）"
                  "精度高，但单次模拟耗时长、计算成本高，使得场地比选、参数敏感性分析与风险量化等需要上万次"
                  "正演模拟的工作难以开展。近年来，傅里叶神经算子（FNO）等算子学习方法被证明可在保持精度的"
                  "前提下，将CO₂羽流演化预测加速千倍以上。本项目旨在：① 复现并适配U-FNO这一CO₂封存羽流预测的"
                  "前沿代理模型；② 在其上叠加深度集成不确定性量化（UQ），使预测具备校准的区间估计；③ 结合场地"
                  "地质几何与六种泄漏机制，构建可解释的泄漏风险三级（绿/黄/红）预警模型，形成“快速代理—"
                  "不确定性—风险预警”一体化的本科可复现技术基线。", indent=True)

    add_subtitle(doc, "（二）研究内容")
    add_para(doc, "1. U-FNO羽流代理模型复现与适配。基于公开CO₂两相流数据集，复现Wen等（2022）提出的U-FNO，"
                  "目标在气相饱和度预测上达到R²>0.99；将异质渗透率场作为模型输入通道，提升对复杂地质结构的适应性。", indent=True)
    add_para(doc, "2. 深度集成不确定性量化。训练5个U-FNO成员（bootstrap重采样），输出羽流预测的均值与区间估计，"
                  "以预测区间覆盖概率（PICP）与区间平均宽度（MPIW）校准，识别模型“不确定”的高风险区域。", indent=True)
    add_para(doc, "3. 泄漏风险三级预警模型。以羽流场、不确定性场与地质几何（盖层、废弃井筒、断层、封存体边界）"
                  "为输入，对照六种泄漏机制给出绿/黄/红预警及触发原因，将黑盒预测翻译为工程可理解的风险语言。", indent=True)
    add_para(doc, "4. 系统集成与原型演示。将前述模块封装为可交互原型：输入地质与注入参数，输出羽流预测、"
                  "不确定性云图与风险等级，支持场地快速筛查演示。", indent=True)

    add_subtitle(doc, "（三）国内外研究现状和发展动态")
    add_para(doc, "算子学习为CO₂封存快速模拟提供了新范式。Li等（2021）提出傅里叶神经算子（FNO），"
                  "首次将偏微分方程解算子学习为连续函数空间映射；Wen等（2022）将其扩展为U-FNO，在宽范围的"
                  "渗透率/孔隙度非均质与注入情形下，对CO₂气相饱和度与压力积累了比原始FNO与CNN更优的预测，"
                  "并开源了代码与数据集。Yin等（2022）提出FNO4CO2，用傅里叶算子学习CO₂浓度时空演化并用于"
                  "监测反演；Grady等（2023）进一步提出模型并行FNO，将代理模型扩展到十亿级变量的大规模"
                  "多相流问题。在三维与真实场地方面，Wang等（2024）提出结合降维的深度学习工作流预测地质封存"
                  "三维状态变量；Han等（2024）构建地质CO₂封存代理模型并嵌入分层MCMC历史匹配，量化地质不确定性。"
                  "在风险导向方面，Zheng等（2024）用U-Net代理快速评估CO₂封存的地质力学风险。在不确定性方法上，"
                  "Lakshminarayanan等（2017）提出的深度集成是目前最实用、无需改动网络结构的校准区间估计方法。", indent=True)
    add_para(doc, "总体趋势是从卷积/UNet点预测走向算子学习，从单一点预测走向不确定性量化，从“单纯代理”"
                  "走向“风险导向的决策支持”。但将“代理模型 + 不确定性量化 + 可解释风险预警”一体打通、"
                  "且面向本科教学可复现的研究仍较少，本项目即在此缺口上展开。", indent=True)

    # 嵌入六种泄漏机制示意图
    if os.path.exists(DIAG):
        doc.add_paragraph()
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rc = cap.add_run("图1  CO₂地质封存六种泄漏机制剖面示意（编号①~⑥对应正文机制）")
        set_cjk(rc, "宋体"); rc.font.size = Pt(9); rc.bold = True
        doc.add_picture(DIAG, width=Inches(5.6))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_subtitle(doc, "（四）创新点与项目特色")
    add_para(doc, "1. 复现并适配SOTA算子模型：在真实CO₂数据上复现U-FNO，并引入异质渗透率通道，"
                  "提升对复杂地质结构的适应性。", indent=True)
    add_para(doc, "2. 不确定性感知：将深度集成UQ与羽流代理结合，给出校准预测区间，显式标识"
                  "“模型不知道”的区域，避免盲目信任点预测。", indent=True)
    add_para(doc, "3. 可解释三级预警：将预测结果对照六种泄漏机制翻译为绿/黄/红风险语言，"
                  "便于工程与监管决策，提升模型可信度。", indent=True)

    add_subtitle(doc, "（五）技术路线、拟解决的问题及预期成果")
    add_para(doc, "1. 技术路线。数据准备（获取U-FNO公开数据集sg_*.pt）→ 复现U-FNO羽流代理 → "
                  "叠加深度集成UQ（5成员bootstrap）→ 构建三级预警（地质几何+六种机制）→ 集成原型演示。", indent=True)
    add_para(doc, "2. 拟解决问题。① 高保真模拟耗时过长，难以支撑海量正演；② 现有代理多为点预测，"
                  "缺乏不确定性；③ 风险不可解释，工程难以直接采用。本项目分别以“算子代理加速”“深度集成UQ”"
                  "“可解释三级预警”回应上述三点。", indent=True)
    add_para(doc, "3. 预期成果。① 一套开源复现与改进代码；② 一个不确定性感知的CO₂封存风险筛查原型系统；"
                  "③ 完整结题报告；④ 视进展整理为学术论文或参加认定赛事。", indent=True)

    add_subtitle(doc, "（六）项目研究进度安排")
    add_para(doc, "1. 2026年第4季度：文献调研与技术研究，搭建PyTorch环境，下载公开数据集，复现U-FNO基线。", indent=True)
    add_para(doc, "2. 2027年第1—2季度：叠加深度集成UQ，构建三级预警规则引擎，准备中期检查材料。", indent=True)
    add_para(doc, "3. 2027年第3—4季度：系统集成与界面设计，开展功能测试与参数校准，优化预警阈值。", indent=True)
    add_para(doc, "4. 2028年第1—3季度：算法与系统完善，整理技术文档与结题报告，准备验收。", indent=True)

    add_subtitle(doc, "（七）已有基础")
    add_para(doc, "1. 研究积累与已取得的成绩。项目组已搭建CNN/FNO代理基线、深度集成UQ训练脚本与"
                  "泄漏风险三级预警规则引擎（risk_model.py）等原型代码，并完成数据集可行性与结题路径论证，"
                  "对“复现+改进”的技术路线形成清晰认识。", indent=True)
    add_para(doc, "2. 已具备的条件、尚缺少的条件及解决方法。已具备：PyTorch开发环境与上述原型代码、"
                  "相关文献调研。尚缺少：真实数据集的下载与GPU算力（拟依托学校实验室服务器或免费GPU"
                  "云环境解决）；以及指导教师对“复现+UQ”作为结题口径的确认（待与导师沟通确定）。", indent=True)

    # 三、经费预算
    add_heading4(doc, "三、经费预算")
    t3 = doc.add_table(rows=14, cols=5)
    t3.style = "Table Grid"
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    budget = [
        ("开支科目", "预算经费（元）", "主要用途", "阶段下达经费计划（元）", "阶段下达经费计划（元）"),
        ("", "", "", "前半阶段", "后半阶段"),
        ("预算经费总额", "3000.00", "", "1400.00", "1600.00"),
        ("1. 业务费", "1100.00", "", "400.00", "700.00"),
        ("（1）计算、分析、测试费", "", "", "", ""),
        ("（2）能源动力费", "", "", "", ""),
        ("（3）会议、差旅费", "800.00", "调研、竞赛", "400.00", "400.00"),
        ("（4）文献检索费", "", "", "", ""),
        ("（5）论文出版费", "300.00", "软件著作权/版面", "0.00", "300.00"),
        ("2. 仪器设备购置费", "", "", "", ""),
        ("3. 实验装置试制费", "", "", "", ""),
        ("4. 材料费", "1900.00", "复印打印、电子耗材", "1000.00", "900.00"),
        ("学校拨款", "", "", "", ""),
        ("财政拨款", "", "", "", ""),
    ]
    for i, row in enumerate(budget):
        for j, val in enumerate(row):
            set_cell(t3.cell(i, j), val, bold=(i == 0 or row[0] in ("预算经费总额",)), size=9.5,
                     align_center=(j > 0))
    # 修正表头第二行
    set_cell(t3.cell(1, 0), "", size=9.5)
    set_cell(t3.cell(1, 1), "", size=9.5)
    set_cell(t3.cell(1, 2), "", size=9.5)

    # 四~七 签名/意见（空白模板，无个人信息）
    add_heading4(doc, "四、项目组成员签名")
    doc.add_paragraph("（此处由项目组成员与指导教师手工签名，此处留白）")

    add_heading4(doc, "五、指导教师意见")
    doc.add_paragraph("导师（签章）：                       年    月    日")

    add_heading4(doc, "六、院系推荐意见")
    doc.add_paragraph("盖 章：                           年    月    日")

    add_heading4(doc, "七、学校推荐意见")
    doc.add_paragraph("盖 章：                           年    月    日")

    doc.save(OUT_APP)
    print("已生成申请书:", OUT_APP)


# ======================================================================
# 文档二：文献真实性查验清单
# ======================================================================
def build_literature():
    doc = Document()
    style_base(doc)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rt = title.add_run("文献真实性查验清单（供学校文献库复核）")
    set_cjk(rt, "黑体"); rt.bold = True; rt.font.size = Pt(15)

    note = doc.add_paragraph()
    rn = note.add_run("说明：本清单所列文献均已通过出版社官网 / arXiv / 学术索引等公开渠道核实“存在且信息一致”。"
                      "正式结题引用前，请在你校提供的文献库（建议优先 Web of Science、Engineering Village、"
                      "CNKI/知网外文库）按下方“检索关键词”复核一次，确认可获取全文与准确的卷、期、页码。")
    set_cjk(rn); rn.font.size = Pt(9.5)

    # 表头
    t = doc.add_table(rows=1, cols=6)
    t.style = "Table Grid"
    headers = ["序号", "文献信息（作者. 标题. 出处, 年.）", "真实度", "检索关键词（中/英）",
               "推荐检索入口 / 直查号", "备注"]
    for j, h in enumerate(headers):
        set_cell(t.cell(0, j), h, bold=True, size=9.5, align_center=True)

    rows = [
        ("1",
         "Wen G, Li Z, Azizzadenesheli K, Anandkumar A, Benson SM. U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow. Advances in Water Resources, 2022, 163: 104180.",
         "高",
         "U-FNO；Fourier neural operator multiphase flow；Gege Wen；CO2 saturation prediction",
         "ScienceDirect / DOI: 10.1016/j.advwatres.2022.104180",
         "本项目复现基线，代码 github.com/gegewen/ufno"),
        ("2",
         "Li Z, Kovachki N, Azizzadenesheli K, et al. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021 (arXiv:2010.08895, 2020).",
         "高",
         "Fourier Neural Operator for Parametric Partial Differential Equations；Zongyi Li；ICLR 2021",
         "arXiv:2010.08895 / Google Scholar",
         "FNO 方法奠基文献"),
        ("3",
         "Yin Z, Siahkoohi A, Louboutin M, Herrmann FJ. Learned coupled inversion for carbon sequestration monitoring and forecasting with Fourier neural operators. Int. Meeting for Applied Geoscience & Energy (IMAGE), 2022. DOI:10.1190/image2022-3722848.1.",
         "高",
         "Learned coupled inversion carbon sequestration Fourier neural operators；Ziyi Yin；FNO4CO2；IMAGE 2022",
         "SEG/Digital Library / DOI: 10.1190/image2022-3722848.1",
         "FNO4CO2 开源代码 github.com/slimgroup/FNO4CO2"),
        ("4",
         "Grady TJ, Khan R, Louboutin M, et al. Model-Parallel Fourier Neural Operators as Learned Surrogates for Large-Scale Parametric PDEs. Computers & Geosciences, 2023, 178: 105402.",
         "高",
         "Model-Parallel Fourier Neural Operators；Computers & Geosciences 178；105402；CO2 multiphase",
         "ScienceDirect / DOI: 10.1016/j.cageo.2023.105402",
         "大规模并行 FNO 参考"),
        ("5",
         "Lakshminarayanan B, Pritzel A, Blundell C. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017 (arXiv:1612.01474).",
         "高",
         "Simple and Scalable Predictive Uncertainty Estimation Deep Ensembles；NeurIPS 2017",
         "arXiv:1612.01474 / Google Scholar",
         "本项目 UQ 方法来源"),
        ("6",
         "Wang H, Hosseini SA, Tartakovsky AM, et al. A deep learning-based workflow for fast prediction of 3D state variables in geological carbon storage: A dimension reduction approach. Journal of Hydrology, 2024, 636: 131219.",
         "高",
         "deep learning workflow fast prediction 3D state variables geological carbon storage；Journal of Hydrology 636；131219",
         "ScienceDirect / DOI: 10.1016/j.jhydrol.2024.131219",
         "三维 GCS 代理参考（ORNL）"),
        ("7",
         "Han Y, et al. Surrogate Model for Geological CO2 Storage and Its Use in Hierarchical MCMC History Matching. Advances in Water Resources, 2024, 187: 104678 (arXiv:2308.06341).",
         "高",
         "Surrogate Model Geological CO2 Storage Hierarchical MCMC History Matching；Advances in Water Resources 187；104678",
         "DOI:10.1016/j.advwatres.2024.104678 / arXiv:2308.06341",
         "代理+不确定性/历史匹配"),
        ("8",
         "Zheng F, Jha B, Jafarpour B. A Deep Learning-Based Surrogate Model for Rapid Assessment of Geomechanical Risks in Geologic CO2 Storage. SPE/AAPG/SEG CCUS Conf., 2024. SPE-CCUS-2024-4003166.",
         "高",
         "Deep Learning-Based Surrogate Model Rapid Assessment Geomechanical Risks Geologic CO2 Storage；SPE-CCUS-2024-4003166",
         "OnePetro / DOI: 10.15530/ccus-2024-4003166",
         "风险导向代理，支撑本项目风险模块"),
        ("9",
         "Tariq Z, Feng Z, Hoteit H, et al. TransUNet: Accelerating Multiphase Flow Prediction in Geological Carbon Storage Systems. ADIPEC 2024, SPE. DOI:10.2118/222334-ms.",
         "中-高",
         "TransUNet Accelerating Multiphase Flow Prediction Geological Carbon Storage；ADIPEC 2024；222334-ms",
         "OnePetro / DOI: 10.2118/222334-ms",
         "可作对比方法（R²≈0.998）"),
        ("10",
         "Benson SM, Cole DR. CO2 sequestration in deep sedimentary formations. Elements, 2008, 4(5): 325-331.",
         "高",
         "CO2 sequestration deep sedimentary formations；Elements 4 325；Sally Benson",
         "GeoScienceWorld / DOI: 10.2111/gselements.4.5.325",
         "封存背景经典文献"),
    ]
    for r in rows:
        cells = t.add_row().cells
        for j, val in enumerate(r):
            set_cell(cells[j], val, size=8.5, align_center=(j in (0, 2)))

    doc.add_paragraph()
    add_subtitle(doc, "待重点核验的扩展候选（学校库未收录时勿引用）")
    add_para(doc, "• NRAP（National Risk Assessment Partnership）风险量化框架相关论文（如 Pawar 等，"
                  "International Journal of Greenhouse Gas Control），建议以“NRAP integrated assessment model carbon storage risk”为关键词核验。", indent=True)
    add_para(doc, "• Zingaro 等关于地质碳封存深度学习模型对比的近期综述/基准论文，较新，请以"
                  "“Comparative Study Deep Learning Geological Carbon Sequestration”核验其正式发表状态后再引用。", indent=True)
    add_para(doc, "• NVIDIA PhysicsNeMo 的 reservoir_simulation / carbon sequestration 示例（开源框架，非传统论文），"
                  "可用作方法对标，引用时按其官方文档著录。", indent=True)

    doc.save(OUT_LIT)
    print("已生成文献查验清单:", OUT_LIT)


if __name__ == "__main__":
    build_application()
    build_literature()
