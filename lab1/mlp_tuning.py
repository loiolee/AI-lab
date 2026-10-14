
from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
MAX_FEATURES = 5000
MAX_ITER = 300

# 比较网络结构和alpha
HIDDEN_LAYER_OPTIONS = [
    (50,),
    (100,),
    (100, 50),
]

ALPHA_VALUES = [
    0.0001,
    0.001,
    0.01,
]


def main():

    # 读取数据

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(
            f"训练集缺少必要列，实际列名：{df.columns.tolist()}"
        )

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("数据中存在缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    # 固定训练集与验证集划分

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验四：MLPClassifier 超参数调优")
    print("=" * 65)
    print(f"训练样本数：{len(X_train)}")
    print(f"验证样本数：{len(X_val)}")
    print(f"TF-IDF 特征上限：{MAX_FEATURES}")
    print(f"最大迭代次数：{MAX_ITER}")
    print(f"隐藏层结构：{HIDDEN_LAYER_OPTIONS}")
    print(f"alpha 参数：{ALPHA_VALUES}")

    # 4. TF-IDF 特征提取

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"\n实际特征数：{len(vectorizer.vocabulary_)}")
    print(f"训练矩阵：{X_train_tfidf.shape}")
    print(f"验证矩阵：{X_val_tfidf.shape}")

    # 构造实验配置
    # 共3种网络结构×3种alpha

    configs = [
        (hidden_layers, alpha)
        for hidden_layers in HIDDEN_LAYER_OPTIONS
        for alpha in ALPHA_VALUES
    ]

    results = []
    best_score = -1.0
    best_config = None
    best_prediction = None

    result_path = DATA_DIR / "mlp_tuning_results.csv"

    # 让收敛警告正常显示，便于判断训练状态
    warnings.simplefilter("default", ConvergenceWarning)

    # 逐组训练和评估

    for index, (hidden_layers, alpha) in enumerate(configs, start=1):
        print("\n" + "-" * 65)
        print(
            f"实验 {index}/{len(configs)}："
            f"hidden_layer_sizes={hidden_layers}, alpha={alpha}"
        )

        start_time = perf_counter()

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            alpha=alpha,
            activation="relu",
            solver="adam",
            max_iter=MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
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

        print(f"实际迭代次数：{model.n_iter_}")
        print(f"Accuracy：{accuracy:.4f}")
        print(f"Macro-F1：{macro_f1:.4f}")
        print(f"训练与预测耗时：{elapsed:.2f} 秒")

        results.append({
            "model": "MLPClassifier",
            "hidden_layer_sizes": str(hidden_layers),
            "alpha": alpha,
            "activation": "relu",
            "max_iter": MAX_ITER,
            "early_stopping": True,
            "random_state": RANDOM_STATE,
            "n_iter": model.n_iter_,
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

        if macro_f1 > best_score:
            best_score = macro_f1
            best_config = (hidden_layers, alpha)
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

    print("\n========== MLP 参数比较结果 ==========")
    print(
        results_df[
            [
                "hidden_layer_sizes",
                "alpha",
                "n_iter",
                "accuracy",
                "macro_f1",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )

    print(f"\n最佳网络结构：{best_config[0]}")
    print(f"最佳 alpha：{best_config[1]}")
    print(f"最佳验证集 Macro-F1：{best_score:.4f}")
    print(
        f"最佳验证集 Accuracy："
        f"{accuracy_score(y_val, best_prediction):.4f}"
    )

    # 保存最佳配置的分类报告

    report = classification_report(
        y_val,
        best_prediction,
        digits=4,
        zero_division=0,
    )

    print("\n========== 最佳 MLP 分类报告 ==========")
    print(report)

    with (DATA_DIR / "mlp_best_report.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"最佳网络结构：{best_config[0]}\n")
        f.write(f"最佳 alpha：{best_config[1]}\n")
        f.write(f"最佳 Macro-F1：{best_score:.4f}\n")
        f.write(
            f"最佳 Accuracy："
            f"{accuracy_score(y_val, best_prediction):.4f}\n\n"
        )
        f.write(report)

    # 保存最佳配置的混淆矩阵

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
        DATA_DIR / "mlp_best_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print("\n已保存文件：")
    print("1. mlp_tuning_results.csv")
    print("2. mlp_best_report.txt")
    print("3. mlp_best_confusion_matrix.csv")


if __name__ == "__main__":
    main()
