
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 文件路径与实验配置

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
ALPHA = 1.0


def main():
    # 读取并检查数据

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    required_columns = {"text", "target"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"训练集必须包含 {required_columns}，"
            f"实际列名为：{df.columns.tolist()}"
        )

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("发现缺失文本或标签，请先检查数据。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    print("=" * 60)
    print("实验一：TF-IDF + MultinomialNB")
    print("=" * 60)
    print(f"总样本数：{len(X)}")
    print(f"类别数量：{pd.Series(y).nunique()}")
    print(f"随机种子：{RANDOM_STATE}")

    # 划分训练集与验证集

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\n训练子集样本数：{len(X_train)}")
    print(f"验证集样本数：{len(X_val)}")

    # TF-IDF 特征提取

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"\nTF-IDF 特征数量：{len(vectorizer.vocabulary_)}")
    print(f"训练特征矩阵形状：{X_train_tfidf.shape}")
    print(f"验证特征矩阵形状：{X_val_tfidf.shape}")

    # 训练基线模型

    model = MultinomialNB(alpha=ALPHA)
    model.fit(X_train_tfidf, y_train)

    # 验证集评估

    y_pred = model.predict(X_val_tfidf)

    accuracy = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\n========== 验证集结果 ==========")
    print(f"Accuracy：{accuracy:.4f}")
    print(f"Macro-F1：{macro_f1:.4f}")

    print("\n========== 分类报告 ==========")
    report = classification_report(
        y_val,
        y_pred,
        digits=4,
        zero_division=0,
    )
    print(report)

    # 保存结果

    labels = sorted(pd.Series(y).unique().tolist())

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).rename_axis("真实标签").to_csv(
        DATA_DIR / "baseline_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    pd.DataFrame([{
        "model": "MultinomialNB",
        "alpha": ALPHA,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_features": len(vectorizer.vocabulary_),
    }]).to_csv(
        DATA_DIR / "baseline_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n结果已保存：")
    print("1. baseline_results.csv")
    print("2. baseline_confusion_matrix.csv")
    print("\n本轮仅使用验证集评估，没有使用无标签测试集。")


if __name__ == "__main__":
    main()
