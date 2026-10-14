
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
)

# 配置

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
C_VALUE = 100
MAX_ITER = 2000

NGRAM_OPTIONS = [
    (1, 1),
    (1, 2),
]


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(f"训练集列名不正确：{df.columns.tolist()}")

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("训练集存在缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    # 与此前实验保持一致的划分
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验五：TF-IDF ngram_range 对比")
    print("=" * 65)
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_val)}")
    print(f"固定参数：C={C_VALUE}, max_features={MAX_FEATURES}")

    results = []
    best_macro_f1 = -1.0
    best_config = None
    best_report = None

    for ngram_range in NGRAM_OPTIONS:
        print("\n" + "-" * 65)
        print(f"开始实验：ngram_range={ngram_range}")

        start_time = perf_counter()

        # 每组配置单独拟合，严格避免验证集信息泄露
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=ngram_range,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)

        model = LogisticRegression(
            C=C_VALUE,
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

        print(f"实际特征数：{len(vectorizer.vocabulary_)}")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"训练与预测总耗时：{elapsed:.2f} 秒")

        results.append({
            "ngram_range": str(ngram_range),
            "max_features": MAX_FEATURES,
            "C": C_VALUE,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_features": len(vectorizer.vocabulary_),
            "elapsed_seconds": round(elapsed, 2),
        })

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_config = ngram_range
            best_report = classification_report(
                y_val,
                y_pred,
                digits=4,
                zero_division=0,
            )

    # 保存两组对比结果
    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        DATA_DIR / "tfidf_ngram_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 对比结果 ==========")
    print(results_df.to_string(index=False))

    print(f"\n最佳 ngram_range：{best_config}")
    print(f"最佳 Macro-F1：{best_macro_f1:.4f}")

    with (DATA_DIR / "tfidf_ngram_best_report.txt").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"最佳 ngram_range：{best_config}\n")
        f.write(f"最佳 Macro-F1：{best_macro_f1:.4f}\n\n")
        f.write(best_report)

    print("\n已保存文件：")
    print("1. tfidf_ngram_results.csv")
    print("2. tfidf_ngram_best_report.txt")


if __name__ == "__main__":
    main()
