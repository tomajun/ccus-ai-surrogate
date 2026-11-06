# Kimberlina 1.2 数据使用礼貌确认邮件（CC-BY，署名即可复用）

> 用途：礼貌告知 NETL 数据联系人你的使用意图，并确认**署名/引用方式**。
> 重要更正：Kimberlina 论文（Alumbaugh et al. 2023, *Geoscience Data Journal*）为 **CC-BY 开放获取**，
> 数据按 CC-BY **署名即可合法复用，无需任何单独许可**。"License Not Specified" 只是 EDX 门户元数据，应以期刊 CC-BY 为准。
> 因此这封邮件是**礼貌与稳妥之举，不是使用的前置条件**——即便对方不回，按 CC-BY 规范署名引用即合规，不阻塞项目。
> 收件人：EDX 数据集页列出的 Point of Contact。**公开仓库不留他人邮箱**，需要时上 EDX 数据集页现查。

---

## 一、可直接发送的邮件（英文正文）

**收件人（To）：** EDX 数据集页的 Point of Contact（本档不留邮箱）
**主题（Subject）：** Usage permission for Kimberlina 1.2 CO₂ Saturation Models (EDX DOI 10.18141/1887287)

**正文：**

Dear EDX dataset contact,

I'm [你的姓名], an undergraduate student at [你的学校英文名] in China. I'm working on a small machine-learning project on CO₂ geological storage for a national undergraduate innovation program, and your Kimberlina 1.2 CO₂ Saturation Models on NETL's EDX caught my attention.

What I'd like to do: use part of the 3D CO₂ saturation fields (the "CO₂ Saturation Models," parts 1–3) as training data for a surrogate model that predicts plume migration from injection parameters. I'd take 2D slices from the 3D volumes to match my model's input format. It's a non-commercial, academic-only project — coursework plus a student competition, nothing beyond that.

Your dataset is published under CC-BY (Alumbaugh et al. 2023, Geoscience Data Journal), so I understand I can reuse it with proper citation. I'm writing mainly to confirm the preferred citation/attribution format and to flag my intended use, in case there's anything specific your team would like me to keep in mind. I'll cite your 2023 paper and the EDX record (DOI 10.18141/1887287) in whatever I submit.

I know these requests take time, so no rush — even a one-line yes, or a pointer to the right person, would help a lot. Thanks for putting this dataset out there; it's exactly the kind of resource students like me rarely get access to.

Best,
[你的姓名]
[你的学校]
[你的邮箱]
[可选：指导教师姓名]

---

## 二、中文逐段说明（帮你理解每句在说什么，不用发）

- **第一段**：自报家门——谁、在哪上学、在做什么项目，并点出是对方数据集的具体内容吸引了你（比"我对贵数据集很感兴趣"更具体、更像真人）。
- **第二段**：具体说清楚你要用什么、怎么用（取 3D 饱和场的 2D 切片喂模型）、明确"非商业、仅学术、课程+比赛"。**越具体，对方越容易放心给许可。**
- **第三段**：直说许可字段是 "Not Specified"，所以主动来问——这是关键，表现出你尊重版权、不是想偷偷用。**并承诺引用论文和 EDX 记录。**
- **第四段**：降低对方心理负担——"不急、回一句就行"。真诚且不给对方压力，反而更容易收到回复。
- **第五段**：谢意具体（"学生很少能拿到这种资源"），不是套话结尾。

---

## 三、发送前你需要填的空（方括号部分）

| 占位 | 填什么 | 示例 |
|---|---|---|
| [你的姓名] | 你的真实姓名（拼音或汉字均可，建议拼音） | Zhang San |
| [你的学校英文名] | 学校官方英文名 | China University of Petroleum |
| [你的学校] | 同上或中文 | 中国石油大学 |
| [你的邮箱] | **用学校邮箱**（edu.cn / 学校域名），别用 QQ/163 | zhangsan@xxx.edu.cn |
| [可选：指导教师姓名] | 若有导师，加上更有分量 | Prof. Li |

---

## 四、详细操作流程（一步一步）

1. **准备邮箱**：用你学校的官方邮箱（带 edu.cn 或学校域名的）。学术邮箱回复率远高于私人邮箱，也显得正规。
2. **打开写信界面**：登录学校邮箱网页或客户端，新建邮件。
3. **填收件人**：到 EDX 数据集页复制 Point of Contact 邮箱（本档不留地址，别手打别猜）。
4. **填主题**：复制上面的 Subject 整行。
5. **填正文**：复制上面的英文正文，把方括号 `[...]` 全部替换成你的真实信息。
6. **检查一遍**：确认没有残留的 `[` 和 `]`；确认收件人拼写正确。
7. **发送**：点发送。**数据本身按 CC-BY 署名即可用**，你可以先把数据下下来"看看长什么样"。为稳妥，建议等对方确认署名方式后再正式写进申报书；但即便无回复，按 CC-BY 规范署名引用即合规，不阻塞使用。
8. **等待与跟进**：对方在美国政府实验室，回复可能要 1–3 周。若 **10 个工作日**后无回复，可发一封简短跟进邮件（附在下方）。
9. **拿到确认后（或即便无回复，按 CC-BY 署名即可）**：保存好引用信息，去 EDX **登录账号**下载 CO₂ Saturation Models part1 的 zip（内含 sim 文件夹、每个含 33 个时间步 csv.zip），用 `sim/kimberlina_csv_to_npz.py convert --src <解压目录>` 转成 .npz，再用 `sim/verify_data_anchor.py kimberlina --src <输出目录>` 汇入。
10. **若被拒或无回复**：**都不阻塞项目**——按 `docs/可用数据集.md` 第 0 节，合成器 + 已接通的 NATCARB 先验 + 可选 PorePy 已足够，Kimberlina 只是加分项；且本仓库已确认 CC-BY 下署名即可用。

---

## 五、跟进邮件（久未回复时用）

**Subject：** Follow-up: usage permission for Kimberlina 1.2 CO₂ Saturation Models

Dear EDX dataset contact,

I wrote to you a couple of weeks ago about using a subset of the Kimberlina 1.2 CO₂ Saturation Models for an academic undergraduate project. I completely understand if it slipped through — just wanted to gently check whether this use would be okay, or whether someone else is the right person to ask.

Either way, thank you for your time.

Best,
[你的姓名]
[你的学校]
[你的邮箱]

---

## 六、为什么这封邮件不像 AI 写的（改动要点）

- 没用 "I hope this email finds you well" 这类聊天机器人开场白。
- 没堆 "delve / robust / leverage / crucial" 等 AI 高频词。
- 没有三段式排比、没有过度客套的套话结尾（如 "The future looks bright"）。
- 用了具体细节（parts 1–3、2D slices、national undergraduate innovation program），让来信显得真实可追溯。
- 语气有起伏：先具体说明，再主动降低对方压力（"no rush"），最后谢意具体——读起来像真学生，不像模板。
- 关键诚信动作：主动说明 CC-BY 并确认署名方式，比闷头用更稳妥；同时让对方知道你尊重署名规范。
