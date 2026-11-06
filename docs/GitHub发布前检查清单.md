# GitHub 发布前检查清单

> 状态（2026-11-06 更新）：已执行 `git init -b main`，**尚未做任何 commit，尚未接远端**。初始化只做预览：`git status --short --untracked-files=all` 共 **109 个待入库文件**，最大单文件 349 KB（`experiments/ufno_uq_report_500.json`），无 >5 MB 文件，全部远低于 GitHub 100 MB 硬上限。
> 下面按"阻断项 → 隐私项 → 质量项 → 执行步骤"排列。标 **决策** 的必须你拍板，我不能替你选。
> 撤销这次初始化：`rm -rf .git`（不影响任何工作文件）。

---

## 一、阻断项（不解决就别 publish）

### 1. 上游代码许可与我们的 MIT 冲突 ⚠️ 决策

`repro/ufno/` 是 `github.com/gegewen/ufno` 的**完整 clone**（固定在 commit `8315fd7b5bd75282b7efe42ee6b8de86543d13cc`），其中 `ufno.py`、`lploss.py`、四个 `.ipynb`、`LICENSE`、`README.md` 都是上游文件，上游许可是：

```
Attribution-NonCommercial-NoDerivatives 4.0 International  (CC BY-NC-ND 4.0)
```

三条限制各自打在我们身上：
- **NC（非商业）**：整仓以 MIT（可商用）发布不兼容。
- **ND（禁止演绎）**：我们**转录并改动**了上游 notebook，产出 `eval_ufno_gas_saturation.py`、`train_ufno_gas_saturation.py`；这些是演绎作品，ND 下不允许再发布。
- **BY（署名）**：这条容易满足，但要求明确标注来源与许可。

可选处理（三选一）：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 不发布上游代码（**推荐**） | 公开仓库里删掉 `repro/ufno/` 的上游文件，只保留我们自己的 `ufno_reader.py` 与两个脚本；README 写"请先 `git clone https://github.com/gegewen/ufno` 并按说明放到 `repro/ufno/`" | 别人多一步手动操作；我们的脚本依赖上游类名，需说明放置路径 |
| B. 改整仓许可为 NC-ND 兼容 | 保留逐字上游文件 + 明确分区，把 `LICENSE` 换成"上游 CC BY-NC-ND 4.0 + 本项目自有代码同条款"，并在 NOTICE 里逐文件标注归属 | 我们自己的代码也失去 MIT 的宽松性；演绎问题仍需向作者申请 |
| C. 先申请许可 | 给 Wen 等作者发邮件（项目里已有 `docs/archive/kimberlina_license_request.md` 的邮件模板经验），拿到书面允许再发 | 时间不可控，可能无回音 |

补充：`数据集/sg_*.pt` 来自同一篇论文，**本来就不入库**（31 GB，且许可未确认），这条我们已经做对了。

### 2. 嵌套 `.git` 会让别人 clone 到空目录 ⚠️ 决策

`repro/ufno/.git` 是真实的上游仓库。外层 `git add .` 会把它记成 gitlink（submodule 引用），但没有 `.gitmodules` → 协作者 clone 后 `repro/ufno/` 是空的。
处理：选方案 A 就顺手删掉整个 `repro/ufno/`（上游文件 + 嵌套 `.git`）；选 B/C 则删 `repro/ufno/.git` 目录，把文件作为 vendored 副本入库，并在 NOTICE 里写死 commit hash。
**删除动作我不会自己做**，需要你说"删"。

### 3. 130 MB 文件会被 GitHub 直接拒绝

GitHub 单文件硬上限 100 MB：`experiments/ufno_pred.npz` = **130 MB**（主线 B 早期产物，现在没人读它）。
另外 `experiments/ufno_uq_500.npz`、`ufno_uq.npz` 各 29 MB，允许但让仓库臃肿，且 `ufno_uq_500.npz` 是 Streamlit 演示的输入。
处理：全部加入 `.gitignore`（已加），演示数据改为**一条命令再生成**：

```bat
.venv\Scripts\python.exe tools\run_uq_risk.py --model_path 数据集\sg_UFNO_4500_trained.pt --data_dir 数据集 --limit 500 --replicas 8 --save_fields 10 --out_npz experiments\ufno_uq_500.npz
```

如果你希望"clone 完就能跑网页"，替代方案是 Git LFS（免费 1 GB 存储/带宽，超额收费）——**决策**：要不要用 LFS。

---

## 二、隐私项（发布即泄露，必须逐条清）

| 位置 | 内容 | 处理 |
|---|---|---|
| `repro/ufno/.git/logs/*` | 含本机用户名与路径（reflog） | 随嵌套 `.git` 一起消失 |
| `docs/archive/kimberlina_license_request.md`、`docs/archive/可用数据集.md` | 原含第三方姓名与邮箱 | ✅ 2026-11-06 已脱敏为"EDX 数据集页列出的 POC"，公开仓库不留他人邮箱；需要时上 EDX 数据集页现查 |
| `docs/*.docx`、`docs/*.pptx` | 个人信息是 `【】` 占位（申请书已核对） | 发布前**人工再翻一遍**组内介绍 docx 与两份 PPT 的封面/收口页，确认没填过真名 |
| `docProps` 元数据 | 2026-11-06 已实测：三个 docx 的 `dc:creator` = `python-docx`、时间戳沿用模板的 2013-12-23，两份 pptx **没有 docProps** | 这一项干净，无需处理；但如果以后用 Word 手工改过这些文件，要重新查一次 `cp:lastModifiedBy` |
| `experiments_smoke.json` | 仓库根目录，里面是 `mean_rel_l2 = 2.8489` —— 就是 `docs/代码级理解指南.md` 坑清单里"按 C 序误读 `sg_test_u.pt`"那次冒烟测试的**坏指标**，且带 `C:/workplace/...` 绝对路径 | 外人会把它当成我们的结果。**建议删除或移进 `docs/archive/` 并加一句失效说明**（删除动作等你说） |
| `tools/finish_cuda_install.sh:5-7`、`tools/run_finish_cuda.sh:6`、`tools/run_local_pipeline.bat:5` | 含 `C:/workplace/ccus-ai-surrogate` 这类**通用**绝对路径，不含用户名 | 不泄露身份，但换机器就跑不了；建议改成 `%~dp0` / `$(cd "$(dirname "$0")/.." && pwd)` 相对定位 |
| `.venv/`、`__pycache__/`、`.pytest_cache/` | 环境缓存，含本机路径 | 已在 `.gitignore` |
| `experiments/_run*.log`、`tools/_*.log` | 运行日志；`tools/_fix_venv.log` 实测含 `C:\Users\<本机用户名>\...` 本机用户名与路径 | 已加入 ignore，`git status` 里确认不到它们 |

另外：仓库里**没有任何密钥类文件**（无 `.env`、无 token），这条是干净的。

---

## 三、质量项（决定别人怎么看这个项目）

本轮状态：1、2、3、4 已完成，5、6、7 待你拍板。

1. ✅ **README 已按当前交付重写**（2026-11-06）。新版叙事：官方 U-FNO 权重 + 零训练 + 输入扰动 UQ + 三轴风险筛查，含"三个关键取舍""结果指标表""已知边界""文档索引"。旧版整份存到 `docs/archive/README_v1_合成数据路线版.md`，并在归档说明的失效表里加了一行。
2. ✅ **申报书承诺的 R² 已落盘**。固化脚本 `tools/r2_check.py` → `experiments/ufno_r2_check.json`（100 场地，点预测，掩膜内）：均值 **0.9932**、中位 0.9963、**82.0% 场地 >0.99**、最低 0.9549；羽流区（`S_g>0.01`）均值 0.9765、最低 0.8184；同批 rel L2 均值 0.0683。README §5 已引用，并注明"点预测，非 MC 均值"。
   → 口径提醒：申报书写的是"R²>0.99"，**平均值口径达标，逐场地口径不成立**（18% 场地 <0.99）。答辩讲"平均 0.993、82% 场地过 0.99"，不要讲"所有场地都过 0.99"。
3. ✅ **`requirements-lock.txt` 已生成**（torch 2.15.0 / numpy 2.5.4 / matplotlib 3.11.2 / scikit-learn 1.9.1 / streamlit 1.66.0 / pandas 3.0.6 / pytest 9.1.1），文件头写明 CUDA 版 torch 需要 `--index-url .../whl/cu132`。Python 版本写在注释里而不是 requirements 里（`python==` 不是合法依赖行，pip 会直接报错）。
4. ✅ **测试已实跑**：`pytest tests/test_uq_risk.py -q` → **23 passed in 0.91 s**，不需要 GPU、不需要数据集。README §4 已明说这句话。
5. **缺 `CHANGELOG`**。"已知边界"这节已经写进 README §7 了（机制 2/3 无几何输入、逐场地覆盖长尾 12.4%、时间轴不可外推、泛化边界=训练参数分布、不宣称加速倍数），CHANGELOG 要不要补是**决策**。
6. `.gitignore` 现在覆盖：`数据集/`、`data/*.npz`、`models/*.pt`、`experiments/*.npz`、`experiments/*.pt`、`experiments/_*.log`、`tools/_*.log`、`docs/_*.txt`、`__pycache__/`、`*.pyc`、`.ipynb_checkpoints/`、`.venv/`、`.env`、`.pytest_cache/`、`results/*.png|pdf`。
   **本轮实测纠正**：gitignore 的 `*` 不跨 `/`，所以 `results/*.png` **管不到** `results/ufno_uq/*.png` —— `git status` 确认 9 张真实数据图 + 2 张结构图（合计约 1.7 MB）**会被入库**。这恰好是"图入库、数据不入库"，README 能直接嵌图；如果你认可就维持现状，不用改 ignore（**决策，默认不动**）。
7. `repro/ufno/` 在 `git status` 里显示为**单条 `repro/ufno/`**，`git add` 会给出 "adding embedded git repository" 警告并记成 gitlink。这正是阻断项 2 的现场证据，clone 方拿到的是空目录。

---

## 四、执行步骤

已完成（本轮）：

```bat
git init -b main                  :: 只初始化，未 commit；撤销用 rm -rf .git
git status --short --untracked-files=all      :: 109 个待入库文件，逐条查过
git add --dry-run repro/ufno                  :: 触发 embedded git repository 警告，证据见 §三.7
```

待你拍板后再做：

```bat
git add .gitignore README.md LICENSE CONTRIBUTING.md requirements.txt requirements-hifi.txt requirements-lock.txt conftest.py
git add src tools tests sim docs data results experiments conftest.py
git status --short            :: 逐行确认没有 数据集/、*.npz、*.log
git commit -m "..."
```

然后：

1. `git log --stat` 自查一次，再用 `git ls-files | xargs grep -l "<本机用户名>\|C:\\Users"` 做**提交后**的泄露复查（这一步能抓出漏网的二进制转文本）。本轮已用同样口径扫过待入库集合：**命中 0 个 `<本机用户名>` / `C:\Users`**，只有第三方邮箱（在 `docs/archive/`）和通用 `C:/workplace` 路径。
2. 建远端仓库、push —— **这两步我不会自己执行**，需要你明确说"推"，并且建议先推到一个 private 仓库看一眼渲染效果再转 public。
3. 发布后 README 里补：上游引用（Wen et al. 2022 + `gegewen/ufno` commit `8315fd7b5bd75282b7efe42ee6b8de86543d13cc`）、数据集获取方式、许可声明。README §10 已按"上游 CC BY-NC-ND 4.0、再分发方式确认中"写了过渡口径，选完 A/B/C 要回来改这一段。

---

## 五、还差三个决策（git init 我已经做了，commit 没做）

- 上游代码走 **A / B / C** 哪个方案？（A = 不发布上游文件，推荐）
- 演示用 npz：**不入库 + 一条命令再生成**（README §4 已经是这个写法），还是 **Git LFS**？
- `results/ufno_uq/*.png`：**照现状入库**（图 1.7 MB，README 能嵌图，推荐），还是改 ignore 不入库？
- 补一条：**首次 commit 现在做吗**（只本地 commit，不 push）？以及 `docs/archive/` 里那两处第三方邮箱要不要脱敏？
