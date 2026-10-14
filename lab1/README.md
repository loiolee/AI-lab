# 当代人工智能实验课：实验一 - 文本分类

最终的实验代码是：final_train_predict.py

## 1. 项目简介

本项目针对多类别文本分类任务，比较 MultinomialNB、线性 SVC、LogisticRegression 和 MLPClassifier 四种算法，并探究 TF-IDF 特征数量、词组范围、模型超参数及文本预处理方式对分类性能的影响。最终使用逻辑回归对无标签测试集生成预测文件。

数据规模：
- 有标签训练数据：7,368 条文本，10 个类别（标签 0–9）。
- 内部验证集：从有标签数据中按 8:2 分层划分，得到 1,474 条验证样本；训练子集为 5,894 条。
- 无标签测试数据：2,457 条文本。

## 2. 环境

本次实验实际运行环境：

- Python 3.14.6
- NumPy 2.4.6
- Pandas 3.0.3
- Scikit-learn 1.9.0


## 3. 代码文件及其在实验中的作用

| 文件 | 在实验中的作用 |
|---|---|
| `inspect_data.py` | 检查 CSV 结构、样本量、标签分布、缺失值、重复文本和预测模板格式。 |
| `baseline_experiment.py` | 建立 TF-IDF（5,000 特征）+ MultinomialNB 基线，计算 Accuracy、Macro-F1，并保存混淆矩阵。 |
| `svc_tuning.py` | 使用线性核 SVC，比较 `C=0.1, 1, 10, 100`。 |
| `logistic_tuning.py` | 在 5,000 个 TF-IDF 特征下比较逻辑回归的 C 参数。 |
| `mlp_tuning.py` | 比较 MLP 的隐藏层结构和 `alpha` 正则化参数。 |
| `tfidf_ngram_experiment.py` | 比较单词特征 `(1, 1)` 与单词加二元词组 `(1, 2)`。 |
| `tfidf_feature_experiment.py` | 比较 5,000、10,000 和 20,000 个 TF-IDF 特征。 |
| `logistic_c_finetuning.py` | 在 20,000 个单词特征下比较逻辑回归的 `C=10, 100, 300, 1000`。 |
| `model_comparison_20000.py` | 在相同的 20,000 维 TF-IDF 特征和数据划分下比较逻辑回归、线性 SVC 与 MLP。 |
| `preprocessing_experiment.py` | 比较原始文本、清理部分邮件头、进一步删除引用行与签名的效果。 |
| `final_train_predict.py` | 使用选定的最终配置在全部有标签数据上重新拟合 TF-IDF 和逻辑回归，并生成测试集预测。 |
| `draft_main.py` | 最初提供的示例代码，仅用于说明实验起点；使用的是 5,000 特征和默认 `SVC()`，不应作为最终预测入口。 |

其中 `final_train_predict.py` 是生成最终预测文件的主入口。注意不要在最终预测完成后运行旧版 `draft_main.py`，因为它可能覆盖预测文件。

## 4. 实验探究流程

本项目的实验步骤如下：
其中所需要的代码执行过程已在上述代码在实验中的作用给出
1. **数据检查：** 确认训练集和测试集的字段、样本数量、标签分布及数据质量。
2. **建立基线：** 使用 TF-IDF（5,000 特征）和 MultinomialNB，获得最初的验证集参照。
3. **模型调参：** 分别测试 SVC、逻辑回归和 MLP 的关键超参数，并记录验证集 Accuracy、Macro-F1 和运行时间。
4. **特征优化：** 比较不同词组范围和特征数量，并进一步细调逻辑回归的 C 值。
5. **统一条件比较：** 将逻辑回归、线性 SVC 和 MLP 放在相同的 20,000 个 TF-IDF 特征下重新评估。
6. **预处理消融：** 比较原始文本与不同清理策略，判断去除邮件头、引用及签名是否改善性能。
7. **最终训练与预测：** 锁定参数后，用全部 7,368 条有标签文本重新拟合向量化器和模型，再预测 2,457 条无标签测试文本。

所有实验的内部训练/验证划分固定使用 `random_state=42` 和分层抽样。每次 TF-IDF 都只在相应训练子集上 `fit`，验证集只执行 `transform`。无标签测试集不参与模型选择和调参。

## 5. 主要实验结果

在统一的 20,000 个 TF-IDF 单词特征下，模型比较结果如下：

| 模型 | 验证集 Accuracy | 验证集 Macro-F1 | 训练与预测耗时 |
|---|---:|---:|---:|
| LogisticRegression (`C=100`) | 93.69% | 93.72% | 2.53 秒 |
| MLPClassifier (`hidden_layer_sizes=(100,)`, `alpha=0.0001`) | 93.35% | 93.41% | 52.28 秒 |
| 线性 SVC (`C=100`) | 92.33% | 92.39% | 39.18 秒 |

最终选用的参数为：

- 模型：`LogisticRegression`
- `C=100`
- `solver='lbfgs'`
- `max_iter=2000`
- TF-IDF：`max_features=20000`
- `ngram_range=(1, 1)`
- 文本预处理：保留原始文本
- 随机种子：42

特征实验显示，在已测试配置下，特征数量由 5,000 增至 20,000 时，逻辑回归的验证集 Macro-F1 从 91.62% 上升至 93.72%；在 5,000 特征下，加入二元词组没有提升表现。预处理实验中，原始文本优于本实验测试的两种清理方案。

注：MultinomialNB 的 88.67% Macro-F1 来自 5,000 特征的基线实验，未在 20,000 特征下复测，因此不纳入上表的同条件比较。上述指标均为验证集结果，并非无标签测试集的准确率。

## 6. 如何运行
如仅需生成最后的predictions.csv：
仅需直接运行final_train_predict.py

如需重现整个实验探究流程：
在项目根目录中打开终端，安装依赖后按需要运行各实验。例如：

```bash
python inspect_data.py
python baseline_experiment.py
python svc_tuning.py
python logistic_tuning.py
python mlp_tuning.py
python tfidf_ngram_experiment.py
python tfidf_feature_experiment.py
python logistic_c_finetuning.py
python model_comparison_20000.py
python preprocessing_experiment.py
```

最后运行：

```bash
python final_train_predict.py
```
各实验脚本会生成相应的 CSV 结果、分类报告或混淆矩阵，具体以脚本中的输出文件名为准。
