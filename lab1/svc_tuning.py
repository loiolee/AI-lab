
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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
MAX_FEATURES = 5000

C_VALUES = [0.1, 1, 10, 100]


def main():

    # 读取数据

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(
            f"训练集列名不正确：{df.columns.tolist()}"
        )

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("数据中存在缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    # 使用固定种子划分数据
    # 保证与基线实验可比较

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验二：SVC 线性核超参数调优")
    print("=" * 65)
    print(f"训练样本数：{len(X_train)}")
    print(f"验证样本数：{len(X_val)}")
    print(f"TF-IDF 特征上限：{MAX_FEATURES}")
    print(f"待测试的 C 值：{C_VALUES}")

    # TF-IDF
    # 仅在训练子集上拟合

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"实际特征数：{len(vectorizer.vocabulary_)}")
    print(f"训练矩阵：{X_train_tfidf.shape}")
    print(f"验证矩阵：{X_val_tfidf.shape}")

    # 逐个比较 C 值
    # 每完成一组就保存结果，便于检查进度

    results = []
    best_score = -1.0
    best_model = None
    best_c = None
    best_prediction = None

    result_path = DATA_DIR / "svc_tuning_results.csv"

    for C in C_VALUES:
        print("\n" + "-" * 65)
        print(f"开始训练 SVC：kernel='linear', C={C}")
        start_time = perf_counter()

        model = SVC(
            kernel="linear",
            C=C,
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

        row = {
            "model": "SVC",
            "kernel": "linear",
            "C": C,
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "elapsed_seconds": round(elapsed, 2),
        }

        results.append(row)

        # 每轮完成后立即写出当前已有的结果
        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"训练与预测耗时：{elapsed:.2f} 秒")

        if macro_f1 > best_score:
            best_score = macro_f1
            best_model = model
            best_c = C
            best_prediction = y_pred.copy()

    # 输出参数比较表

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== SVC 参数比较结果 ==========")
    print(
        results_df[
            ["C", "accuracy", "macro_f1", "elapsed_seconds"]
        ].to_string(index=False)
    )

    print(f"\n最佳 C：{best_c}")
    print(f"最佳验证集 Macro-F1：{best_score:.4f}")

    # 输出最佳 SVC 的分类报告

    report = classification_report(
        y_val,
        best_prediction,
        digits=4,
        zero_division=0,
    )

    print("\n========== 最佳 SVC 分类报告 ==========")
    print(report)

    report_path = DATA_DIR / "svc_best_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"最佳模型：SVC(kernel='linear', C={best_c})\n")
        f.write(f"Accuracy：{accuracy_score(y_val, best_prediction):.4f}\n")
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
        DATA_DIR / "svc_best_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print("\n已保存文件：")
    print("1. svc_tuning_results.csv")
    print("2. svc_best_report.txt")
    print("3. svc_best_confusion_matrix.csv")


if __name__ == "__main__":
    main()
