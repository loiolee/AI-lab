
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 实验配置

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 1)
C_VALUES = [10, 100, 300, 1000]
MAX_ITER = 2000


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(f"训练集列名不正确：{df.columns.tolist()}")

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("数据中存在缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    # 固定数据划分，确保不同C之间可以比较
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验七：20,000 特征下的逻辑回归 C 参数细调")
    print("=" * 65)
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_val)}")
    print(f"max_features：{MAX_FEATURES}")
    print(f"ngram_range：{NGRAM_RANGE}")
    print(f"C 候选值：{C_VALUES}")

    # 只在训练子集上拟合词表和IDF权重
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"实际特征数：{len(vectorizer.vocabulary_)}")

    results = []
    best_score = -1.0
    best_c = None
    best_accuracy = None
    best_prediction = None

    result_path = DATA_DIR / "logistic_c_finetuning_results.csv"

    for C in C_VALUES:
        print("\n" + "-" * 65)
        print(f"开始训练：LogisticRegression(C={C})")

        start_time = perf_counter()

        model = LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_val_tfidf)

        elapsed = perf_counter() - start_time

        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"训练与预测耗时：{elapsed:.2f} 秒")

        results.append({
            "model": "LogisticRegression",
            "C": C,
            "max_features": MAX_FEATURES,
            "ngram_range": str(NGRAM_RANGE),
            "max_iter": MAX_ITER,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "elapsed_seconds": round(elapsed, 2),
        })

        # 每完成一组参数，立即保存已有结果
        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig",
        )

        if macro_f1 > best_score:
            best_score = macro_f1
            best_c = C
            best_accuracy = accuracy
            best_prediction = y_pred.copy()

    # 输出完整参数比较
    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== C 参数比较结果 ==========")
    print(
        results_df[
            ["C", "accuracy", "macro_f1", "elapsed_seconds"]
        ].to_string(index=False)
    )

    print(f"\n最佳 C：{best_c}")
    print(f"最佳 Accuracy：{best_accuracy:.4f}")
    print(f"最佳 Macro-F1：{best_score:.4f}")

    # 保存最佳参数对应的分类报告
    report = classification_report(
        y_val,
        best_prediction,
        digits=4,
        zero_division=0,
    )

    print("\n========== 最佳模型分类报告 ==========")
    print(report)

    with (DATA_DIR / "logistic_c_best_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"最佳 C：{best_c}\n")
        f.write(f"max_features：{MAX_FEATURES}\n")
        f.write(f"Accuracy：{best_accuracy:.4f}\n")
        f.write(f"Macro-F1：{best_score:.4f}\n\n")
        f.write(report)

    labels = sorted(pd.Series(y).unique().tolist())

    cm = confusion_matrix(
        y_val,
        best_prediction,
        labels=labels,
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    ).rename_axis("真实标签").to_csv(
        DATA_DIR / "logistic_c_best_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print("\n已保存文件：")
    print("1. logistic_c_finetuning_results.csv")
    print("2. logistic_c_best_report.txt")
    print("3. logistic_c_best_confusion_matrix.csv")


if __name__ == "__main__":
    main()
