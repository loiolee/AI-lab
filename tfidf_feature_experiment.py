
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ==================================================
# 1. 实验配置
# ==================================================

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
C_VALUE = 100
MAX_ITER = 2000

FEATURE_OPTIONS = [5000, 10000, 20000]


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

    # 与此前实验保持相同的数据划分
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验六：TF-IDF 特征数量对比")
    print("=" * 65)
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_val)}")
    print(f"C：{C_VALUE}")
    print("ngram_range：(1, 1)")
    print(f"特征数量候选：{FEATURE_OPTIONS}")

    results = []
    result_path = DATA_DIR / "tfidf_feature_results.csv"

    best_macro_f1 = -1.0
    best_features = None

    for max_features in FEATURE_OPTIONS:
        print("\n" + "-" * 65)
        print(f"开始实验：max_features={max_features}")

        start_time = perf_counter()

        # 每组参数均仅在训练子集上拟合 TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
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
        print(f"训练矩阵：{X_train_tfidf.shape}")
        print(f"验证矩阵：{X_val_tfidf.shape}")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"总耗时：{elapsed:.2f} 秒")

        results.append({
            "model": "LogisticRegression",
            "C": C_VALUE,
            "ngram_range": "(1, 1)",
            "max_features": max_features,
            "n_features": len(vectorizer.vocabulary_),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "elapsed_seconds": round(elapsed, 2),
        })

        # 每完成一组就更新结果文件
        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig",
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_features = max_features

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 特征数量比较结果 ==========")
    print(
        results_df[
            [
                "max_features",
                "n_features",
                "accuracy",
                "macro_f1",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )

    print(f"\n最佳特征数量：{best_features}")
    print(f"最佳验证集 Macro-F1：{best_macro_f1:.4f}")
    print(f"结果已保存至：{result_path}")


if __name__ == "__main__":
    main()
