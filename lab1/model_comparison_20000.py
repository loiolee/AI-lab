
from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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

# 读取数据

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"找不到训练文件：{TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(
            f"训练集列名不正确：{df.columns.tolist()}"
        )

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("数据中存在缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    # 固定数据划分

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验八：20,000 个 TF-IDF 特征下的模型比较")
    print("=" * 65)
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_val)}")
    print(f"特征数量上限：{MAX_FEATURES}")

    # 只拟合一次 TF-IDF，所有模型共用同一特征矩阵


    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 1),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"实际特征数：{len(vectorizer.vocabulary_)}")
    print(f"训练矩阵：{X_train_tfidf.shape}")
    print(f"验证矩阵：{X_val_tfidf.shape}")

    # 构建待比较模型


    models = [
        (
            "LogisticRegression",
            LogisticRegression(
                C=100,
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "SVC_linear",
            SVC(
                kernel="linear",
                C=100,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "MLPClassifier",
            MLPClassifier(
                hidden_layer_sizes=(100,),
                alpha=0.0001,
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    results = []
    reports = []
    labels = sorted(pd.Series(y).unique().tolist())

    results_path = (
        DATA_DIR / "model_comparison_20000_results.csv"
    )
    reports_path = (
        DATA_DIR / "model_comparison_20000_reports.txt"
    )

    warnings.simplefilter("default", ConvergenceWarning)

    # 逐个训练并评估

    for name, model in models:
        print("\n" + "-" * 65)
        print(f"开始训练：{name}")

        start_time = perf_counter()

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

        row = {
            "model": name,
            "max_features": MAX_FEATURES,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "elapsed_seconds": round(elapsed, 2),
        }

        if hasattr(model, "n_iter_"):
            row["n_iter"] = str(model.n_iter_)

        results.append(row)

        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"训练与预测耗时：{elapsed:.2f} 秒")

        if hasattr(model, "n_iter_"):
            print(f"迭代次数：{model.n_iter_}")

        report = classification_report(
            y_val,
            y_pred,
            digits=4,
            zero_division=0,
        )

        reports.append(
            f"模型：{name}\n"
            f"Accuracy：{accuracy:.4f}\n"
            f"Macro-F1：{macro_f1:.4f}\n\n"
            f"{report}\n"
        )

        # 保存每个模型的混淆矩阵
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
            DATA_DIR / f"{name}_20000_confusion_matrix.csv",
            encoding="utf-8-sig",
        )

        # 每完成一个模型便保存当前结果
        pd.DataFrame(results).to_csv(
            results_path,
            index=False,
            encoding="utf-8-sig",
        )

        with reports_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(reports))

    # 汇总结果

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 最终模型比较 ==========")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )

    best = results_df.iloc[0]

    print(f"\n当前最佳模型：{best['model']}")
    print(f"最佳 Accuracy：{best['accuracy']:.4f}")
    print(f"最佳 Macro-F1：{best['macro_f1']:.4f}")

    print("\n已保存文件：")
    print("1. model_comparison_20000_results.csv")
    print("2. model_comparison_20000_reports.txt")
    print("3. 三个模型各自的混淆矩阵 CSV")


if __name__ == "__main__":
    main()
