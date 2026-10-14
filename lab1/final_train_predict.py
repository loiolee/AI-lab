
from pathlib import Path
import shutil

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 路径与最终配置

DATA_DIR = Path(__file__).resolve().parent

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data_unlabeled.csv"

# 中间预测文件和正式提交文件
FINAL_PRED_PATH = DATA_DIR / "predictions_final.csv"
SUBMISSION_PATH = DATA_DIR / "predictions.csv"
BACKUP_PATH = DATA_DIR / "predictions_original_backup.csv"

RANDOM_STATE = 42
MAX_FEATURES = 20000
C_VALUE = 100
MAX_ITER = 2000


def main():
    print("=" * 65)
    print("最终模型训练与预测")
    print("=" * 65)

    # 检查并读取数据

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练集：{TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"找不到测试集：{TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if not {"text", "target"}.issubset(train_df.columns):
        raise ValueError(
            f"训练集需要 text 和 target 两列，"
            f"实际列名为：{train_df.columns.tolist()}"
        )

    if "text" not in test_df.columns:
        raise ValueError(
            f"测试集缺少 text 列，实际列名为："
            f"{test_df.columns.tolist()}"
        )

    if train_df["text"].isna().any():
        raise ValueError("训练集存在缺失文本。")

    if train_df["target"].isna().any():
        raise ValueError("训练集存在缺失标签。")

    if test_df["text"].isna().any():
        raise ValueError("测试集存在缺失文本。")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].to_numpy()
    X_test = test_df["text"].astype(str).tolist()

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("训练集或测试集为空。")

    print(f"训练样本数：{len(X_train)}")
    print(f"测试样本数：{len(X_test)}")
    print(f"类别数量：{train_df['target'].nunique()}")

    # 在全部有标签数据上拟合 TF-IDF
    # 测试集只能 transform，不能参与 fit

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 1),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"实际特征数：{len(vectorizer.vocabulary_)}")
    print(f"训练矩阵形状：{X_train_tfidf.shape}")
    print(f"测试矩阵形状：{X_test_tfidf.shape}")

    # 训练最终逻辑回归模型

    model = LogisticRegression(
        C=C_VALUE,
        max_iter=MAX_ITER,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_tfidf, y_train)
    print("\n最终模型训练完成。")

    # 预测测试集

    predictions = model.predict(X_test_tfidf)

    # 检查预测结果

    if len(predictions) != len(test_df):
        raise RuntimeError("预测数量与测试集样本数不一致。")

    if pd.isna(predictions).any():
        raise RuntimeError("预测结果中存在缺失值。")

    valid_labels = set(pd.unique(y_train).tolist())
    predicted_labels = set(predictions.tolist())

    if not predicted_labels.issubset(valid_labels):
        raise RuntimeError("预测标签中出现了训练集未包含的类别。")

    # 先保存独立的最终预测文件
    # 单列、无表头、无行索引

    pd.DataFrame(predictions).to_csv(
        FINAL_PRED_PATH,
        index=False,
        header=False,
        encoding="utf-8",
    )

    # 重新读取文件，检查实际保存的格式
    saved_df = pd.read_csv(
        FINAL_PRED_PATH,
        header=None,
    )

    if saved_df.shape != (len(test_df), 1):
        raise RuntimeError(
            f"预测文件格式异常：{saved_df.shape}"
        )

    if saved_df.isna().any().any():
        raise RuntimeError("保存后的预测文件存在缺失值。")

    saved_labels = set(saved_df.iloc[:, 0].tolist())

    if not saved_labels.issubset(valid_labels):
        raise RuntimeError("保存后的文件包含非法预测标签。")

    # 备份原文件，再更新正式提交文件

    if SUBMISSION_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(SUBMISSION_PATH, BACKUP_PATH)
        print(f"\n原始文件已备份：{BACKUP_PATH.name}")

    shutil.copy2(FINAL_PRED_PATH, SUBMISSION_PATH)

    # 最后再次检查正式提交文件
    submission_df = pd.read_csv(
        SUBMISSION_PATH,
        header=None,
    )

    assert submission_df.shape == (len(test_df), 1)
    assert submission_df.notna().all().all()

    print("\n========== 最终检查 ==========")
    print(f"提交文件：{SUBMISSION_PATH.name}")
    print(f"预测行数：{len(submission_df)}")
    print(f"预测列数：{submission_df.shape[1]}")
    print(f"预测标签：{sorted(saved_labels)}")
    print(f"缺失值数量：{int(submission_df.isna().sum().sum())}")

    print("\n预测类别分布：")
    print(
        submission_df.iloc[:, 0]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # 保存模型参数记录
    config_path = DATA_DIR / "final_model_config.txt"

    with config_path.open("w", encoding="utf-8") as f:
        f.write("最终模型配置\n")
        f.write("Model: LogisticRegression\n")
        f.write(f"C: {C_VALUE}\n")
        f.write(f"solver: lbfgs\n")
        f.write(f"max_iter: {MAX_ITER}\n")
        f.write(f"max_features: {MAX_FEATURES}\n")
        f.write("ngram_range: (1, 1)\n")
        f.write("Preprocessing: raw text\n")
        f.write(f"random_state: {RANDOM_STATE}\n")
        f.write(f"training_samples: {len(X_train)}\n")
        f.write(f"test_samples: {len(X_test)}\n")
        f.write(f"sklearn_version: {sklearn.__version__}\n")

    print("最终预测文件为predictions.csv。")


if __name__ == "__main__":
    main()
