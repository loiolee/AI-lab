
from pathlib import Path
from time import perf_counter
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# 实验配置

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 20000
C_VALUE = 100
MAX_ITER = 2000

HEADER_PATTERN = re.compile(
    r"^(from|subject|organization|lines|reply-to|"
    r"nntp-posting-host|distribution|newsgroups|"
    r"message-id|date|path|x-newsreader|references|"
    r"keywords|sender|followup-to|summary|expires)\s*:",
    re.IGNORECASE,
)


# 预处理函数

def remove_headers_keep_subject(text):
    """删除可识别的邮件头元数据，尽量保留 Subject 和正文。"""

    match = re.search(r"\r?\n[ \t]*\r?\n", text)

    # 无法识别标准头部与正文分隔时，不贸然删除内容
    if match is None:
        return text

    header_block = text[:match.start()]
    body = text[match.end():]

    header_lines = header_block.splitlines()

    # 只有头部区域出现已知字段，才将其视为邮件头
    if not any(
        HEADER_PATTERN.match(line)
        for line in header_lines
    ):
        return text

    # 提取 Subject，并尽量保留其换行续接内容
    subject_lines = []
    collecting_subject = False

    for line in header_lines:
        if re.match(r"^\s*Subject\s*:", line, re.IGNORECASE):
            subject_lines = [line]
            collecting_subject = True

        elif (
            collecting_subject
            and line[:1].isspace()
            and not HEADER_PATTERN.match(line.strip())
        ):
            subject_lines.append(line)

        else:
            collecting_subject = False

    subject = "\n".join(subject_lines)

    if subject:
        return subject + "\n\n" + body

    return body


def remove_quotes_and_signatures(text):
    """删除以 > 开头的引用行及标准签名分隔符之后的内容。"""

    kept_lines = []

    for line in text.splitlines():
        # 常见邮件签名分隔符
        if re.match(r"^\s*--\s*$", line):
            break

        # 删除引用行
        if re.match(r"^\s*>", line):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)


def keep_raw_text(text):
    """原始文本对照组。"""
    return text


def main():
    # 读取数据并固定数据划分

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到训练文件：{TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if not {"text", "target"}.issubset(df.columns):
        raise ValueError(f"训练集列名不正确：{df.columns.tolist()}")

    if df["text"].isna().any() or df["target"].isna().any():
        raise ValueError("发现缺失文本或标签。")

    X = df["text"].astype(str).tolist()
    y = df["target"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("实验九：文本预处理消融实验")
    print("=" * 65)
    print(f"训练样本：{len(X_train)}")
    print(f"验证样本：{len(X_val)}")
    print(f"max_features：{MAX_FEATURES}")
    print(f"C：{C_VALUE}")

    # 每个方案共用完全相同的数据划分
    configurations = [
        ("A_raw", keep_raw_text),
        ("B_keep_subject", remove_headers_keep_subject),
        (
            "C_remove_quotes_signatures",
            lambda text: remove_quotes_and_signatures(
                remove_headers_keep_subject(text)
            ),
        ),
    ]

    results = []
    reports = []

    result_path = DATA_DIR / "preprocessing_results.csv"
    report_path = DATA_DIR / "preprocessing_reports.txt"

    # 逐一比较预处理方案

    for name, transform in configurations:
        print("\n" + "-" * 65)
        print(f"开始实验：{name}")

        start_time = perf_counter()

        # 相同预处理函数分别应用于训练集和验证集
        train_texts = [transform(text) for text in X_train]
        val_texts = [transform(text) for text in X_val]

        # 每个方案独立拟合自己的训练词表
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=(1, 1),
        )

        X_train_tfidf = vectorizer.fit_transform(train_texts)
        X_val_tfidf = vectorizer.transform(val_texts)

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
        print(f"总耗时：{elapsed:.2f} 秒")

        results.append({
            "preprocessing": name,
            "max_features": MAX_FEATURES,
            "C": C_VALUE,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_features": len(vectorizer.vocabulary_),
            "elapsed_seconds": round(elapsed, 2),
        })

        reports.append(
            f"预处理方案：{name}\n"
            f"Accuracy：{accuracy:.4f}\n"
            f"Macro-F1：{macro_f1:.4f}\n\n"
            + classification_report(
                y_val,
                y_pred,
                digits=4,
                zero_division=0,
            )
            + "\n"
        )

        # 每完成一组就保存已获得的结果
        pd.DataFrame(results).to_csv(
            result_path,
            index=False,
            encoding="utf-8-sig",
        )

        with report_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(reports))

    # 汇总结果

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 预处理对比结果 ==========")
    print(
        results_df[
            [
                "preprocessing",
                "accuracy",
                "macro_f1",
                "n_features",
                "elapsed_seconds",
            ]
        ].to_string(index=False)
    )

    best = results_df.iloc[0]

    print(f"\n当前最佳预处理方案：{best['preprocessing']}")
    print(f"最佳 Accuracy：{best['accuracy']:.4f}")
    print(f"最佳 Macro-F1：{best['macro_f1']:.4f}")

    print("\n已保存文件：")
    print("1. preprocessing_results.csv")
    print("2. preprocessing_reports.txt")


if __name__ == "__main__":
    main()
