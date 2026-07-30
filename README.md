# classical-ml-lab

> Reproducible, production-minded classification, clustering, regression, data-quality, and SHAP experiments. Built for learning and engineering review—not for real credit or medical decisions.

一个可复现的经典机器学习实验室：九个默认离线实验，加上三个可选 Boosting 回归实验，展示可信的数据处理、模型训练、评估、数据质量和可审计解释产物。

本项目是已归档的 [`machinelearning-blob`](https://github.com/silasxlx/machinelearning-blob) 教学示例的重新设计版本。

## 是什么、为什么、怎么用

| 问题 | 回答 |
| --- | --- |
| **是什么？** | 一个统一运行经典分类、KMeans 聚类、Ridge/Boosting 回归、数据质量报告和 Tree SHAP 解释的 Python 包与 CLI。 |
| **为什么？** | 把容易“跑出一个分数”的旧 Notebook，升级为路径无关、评估可信、依赖可复现、结果可审计的工程项目。 |
| **怎么用？** | 安装 `uv` 后执行下面三条命令，默认数据全部离线可用。 |

> [!IMPORTANT]
> 本项目仅用于教学和工程演示。默认数据是 scikit-learn 内置数据或本地生成的合成数据。模型不得直接用于真实授信、医疗诊断或其他高风险决策。

## Quick Start

前置条件：[Git](https://git-scm.com/) 与 [uv](https://docs.astral.sh/uv/)；Python 和项目依赖由 `uv` 管理。

<!-- quickstart:start -->
```bash
git clone https://github.com/silasxlx/classical-ml-lab.git
cd classical-ml-lab
uv run ml-lab run all
```
<!-- quickstart:end -->

第三条命令会自动创建环境、安装锁定的核心依赖，并在 `artifacts/<run-id>/` 下生成九个默认实验的指标、数据质量报告和图表。

查看或单独运行实验：

```bash
uv run ml-lab list
uv run ml-lab run svm --seed 42
uv run ml-lab run random-forest --seed 42
uv run ml-lab run logistic-regression --seed 42
uv run ml-lab run knn --seed 42
uv run ml-lab run naive-bayes --seed 42
uv run ml-lab run kmeans --seed 42
uv run ml-lab run ridge-regression --seed 42
```

## 九个核心实验

| 实验 | 默认数据 | 关键工程点 | 主要产物 |
| --- | --- | --- | --- |
| Decision Tree | Iris | 固定划分、无需 Graphviz、现代 `plot_tree` | 多分类指标、树图 |
| Random Forest | 本地生成的不平衡数据 | 插补置于 Pipeline、训练集内分层 CV | 二分类指标、CV 摘要、特征重要性 |
| AdaBoost | Breast Cancer Wisconsin | 当前 `estimator` API、独立测试集 | 二分类指标、特征重要性 |
| SVM | Iris 二分类子集 | `StandardScaler + SVC` Pipeline | 二分类指标、决策边界 |
| Logistic Regression | 本地生成的不平衡数据 | `Imputer + Scaler + LogisticRegression`、训练集内分层 CV | 二分类指标、CV 摘要、标准化系数图 |
| KNN | Iris | `StandardScaler + KNeighborsClassifier` | 多分类指标、混淆矩阵图 |
| Naive Bayes | Iris | 固定 GaussianNB、独立测试集 | 多分类指标、混淆矩阵图 |
| KMeans | Iris 特征 | 目标标签不参与拟合、独立聚类 Schema | 聚类指标、簇大小、二维投影图 |
| Ridge Regression | Diabetes | `StandardScaler + Ridge`、独立回归 holdout | MAE、RMSE、R²、预测与残差图 |

二分类实验报告 accuracy、balanced accuracy、precision、recall、F1、ROC-AUC、PR-AUC 和混淆矩阵。Iris 多分类实验同时报告 macro/weighted 指标和 OvR AUC。KMeans 单独报告 silhouette、ARI、NMI、inertia 和簇大小。回归实验报告 MAE、RMSE 和 R²，不把不同任务伪装成同一种结果。

每个核心实验都会生成聚合式 `data-quality.json` 和 `figures/data-quality.png`，记录类型、缺失、非有限值、重复、常量、分位数与 IQR 异常值计数；它不会保存原始行、本机绝对路径或患者记录。

## 可选 Boosting 与 SHAP

三个 Boosting 回归实验不属于核心 `run all`，需要显式安装对应 extra：

```bash
uv run --extra xgboost ml-lab run xgboost-regression
uv run --extra lightgbm ml-lab run lightgbm-regression
uv run --extra catboost ml-lab run catboost-regression
uv run --all-extras ml-lab run boosting-all
```

它们复用 Diabetes 数据、同一 holdout、回归指标和数据质量契约，并额外生成全局/局部 Tree SHAP 图与脱敏 `explanations.json`。SHAP 只描述模型在给定数据上的行为，不代表因果关系、医学依据或临床安全性。

## 结果预览

以下图片由仓库代码在 `seed=42` 下实际生成，分别展示无监督聚类、回归诊断、数据质量和模型解释产物。SHAP 仅描述模型行为，不代表因果关系。

| KMeans 聚类结果 | Ridge 回归诊断 |
| --- | --- |
| ![KMeans 在 Iris 花瓣特征上的预测簇与聚类中心](docs/assets/preview-kmeans-clusters.png) | ![Ridge 回归的真实值、预测值与残差诊断](docs/assets/preview-ridge-predictions.png) |
| **数据质量报告** | **XGBoost 全局 SHAP** |
| ![特征完整性与目标分布数据质量报告](docs/assets/preview-data-quality.png) | ![XGBoost 特征的全局平均绝对 SHAP 值](docs/assets/preview-shap-global.png) |

## 输出与可审计性

```text
artifacts/<run-id>/
├── run.json
└── experiments/
    ├── decision-tree/
    │   ├── metrics.json
    │   ├── data-quality.json
    │   └── figures/
    ├── random-forest/
    ├── adaboost/
    ├── svm/
    ├── logistic-regression/
    ├── knn/
    ├── naive-bayes/
    ├── kmeans/
    └── ridge-regression/
```

- `run.json` 记录 seed、配置哈希、依赖版本、运行状态和相对产物路径。
- `metrics.json` 记录数据来源/指纹、训练测试划分、模型配置、指标、混淆矩阵和 CV 摘要。
- `data-quality.json` 记录聚合数据质量事实和 warning，不自动删除异常值。
- 可选树模型的 `explanations.json` 记录特征顺序、SHAP 值、基值、模型输出和加法残差，不记录完整样本。
- 重复执行不会静默覆盖旧产物。
- 相同数据、配置、依赖与 seed 的指标在同一环境中可复现。

JSON Schema 位于 [`schemas/`](schemas/)，公共接口见 [`docs/api.md`](docs/api.md)。

## 架构

```mermaid
flowchart LR
    CLI["CLI<br/>ml-lab"] --> Runner["Runner<br/>参数校验 · 运行编排"]
    Runner --> Registry["实验注册表<br/>稳定顺序 · 统一入口"]

    Data["离线数据<br/>内置 · 合成 · 本地 CSV"] --> Classification
    Data --> Clustering
    Data --> Regression
    Data --> DataQuality["聚合数据质量报告"]

    Registry --> Classification["分类实验<br/>7 experiments"]
    Registry --> Clustering["聚类实验<br/>KMeans"]
    Registry --> Regression["回归实验<br/>Ridge · optional Boosting"]

    Classification --> ClassificationMetrics["分类指标<br/>Holdout · CV · AUC"]
    Clustering --> ClusteringMetrics["聚类指标<br/>Silhouette · ARI · NMI"]
    Regression --> RegressionMetrics["回归指标<br/>MAE · RMSE · R²"]
    Regression --> Explainability["Tree SHAP<br/>training-only background"]

    ClassificationMetrics --> Artifacts["可审计产物<br/>JSON · PNG · schema"]
    ClusteringMetrics --> Artifacts
    RegressionMetrics --> Artifacts
    DataQuality --> Artifacts["可审计产物<br/>JSON · PNG · schema"]
    Explainability --> Artifacts
```

核心代码位于 `src/classical_ml_lab`：

- `data.py`：内置/合成数据及可选信用 CSV 校验。
- `metrics.py`：统一分类与回归指标。
- `data_quality.py`：跨任务的聚合数据质量报告。
- `experiments/`：每个算法的训练、评估和图表。
- `runner.py`、`artifacts.py`：运行编排、哈希、防覆盖和审计产物。
- `cli.py`：稳定的命令、参数与退出码。

架构、数据边界和评估设计见 [`docs/design.md`](docs/design.md)。

## 使用本地信用数据（可选）

大型原始信用数据不进入仓库，也不会被自动下载。Random Forest 与 Logistic Regression 共用同一套本地数据校验；只有在你已合法取得数据并接受其来源条款时，才可显式传入本地文件：

```bash
uv run ml-lab run logistic-regression \
  --dataset credit \
  --data-path data/raw/cs-training.csv
```

加载器会在训练前校验字段、目标、类型、样本数和类别分布。详细说明见 [`docs/design.md`](docs/design.md)。即使使用该数据，结果仍然只是教学实验，不能作为真实授信决策依据。

## 开发与质量

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run mypy src
uv run pytest --cov=classical_ml_lab --cov-branch --cov-fail-under=85
```

可选 Boosting 专项开发环境使用 `uv sync --locked --all-extras`。

CI 覆盖 Python 3.11–3.13、Ubuntu 和 Windows，并执行 lint、类型检查、测试、Notebook smoke test 与安全扫描。核心逻辑覆盖率目标为 85%，不以追求 100% 数字替代有效行为断言。

## 文档

- [公共 API 与 CLI](docs/api.md)
- [架构、数据与评估设计](docs/design.md)
- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [变更记录](CHANGELOG.md)

## 贡献

欢迎修正文档、增加边界测试、改善可解释图表或提出新的经典算法实验。请先阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，使用 Conventional Commits，并确保 lint、类型检查和测试通过。

项目采用 [MIT License](LICENSE)。
