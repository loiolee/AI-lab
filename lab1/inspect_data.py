from pathlib import Path
import pandas as pd

# 获取当前脚本所在目录
DATA_DIR = Path(__file__).resolve().parent

train_path = DATA_DIR / "train_data.csv"
test_path = DATA_DIR / "test_data_unlabeled.csv"
pred_path = DATA_DIR / "predictions.csv"


def text_statistics(series):
    """统计文本长度和词数，不修改原始数据。"""
    texts = series.fillna("").astype(str)

    char_lengths = texts.str.len()
    word_counts = texts.str.split().str.len()

    return {
        "缺失文本数": int(series.isna().sum()),
        "空文本数": int(texts.str.strip().eq("").sum()),
        "最短字符数": int(char_lengths.min()),
        "平均字符数": round(float(char_lengths.mean()), 2),
        "最长字符数": int(char_lengths.max()),
        "平均词数": round(float(word_counts.mean()), 2),
    }


def main():
    # 1. 检查文件是否存在
    for path in (train_path, test_path, pred_path):
        print(f"{path.name}: {'存在' if path.exists() else '不存在'}")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "缺少训练集或测试集，请确认三个文件所在目录。"
        )

    # 2. 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("\n========== 训练集 ==========")
    print("数据形状:", train_df.shape)
    print("列名:", train_df.columns.tolist())
    print("前五行:")
    print(train_df.head().to_string(index=False))

    print("\n========== 测试集 ==========")
    print("数据形状:", test_df.shape)
    print("列名:", test_df.columns.tolist())
    print("前五行:")
    print(test_df.head().to_string(index=False))

    # 3. 检查必要的列
    required_train = {"text", "target"}
    required_test = {"text"}

    if not required_train.issubset(train_df.columns):
        raise ValueError(
            f"训练集缺少必要列: {required_train - set(train_df.columns)}"
        )

    if not required_test.issubset(test_df.columns):
        raise ValueError(
            f"测试集缺少必要列: {required_test - set(test_df.columns)}"
        )

    # 4. 检查标签数量与类别分布
    print("\n========== 标签分布 ==========")
    print(train_df["target"].value_counts(dropna=False).sort_index())
    print("类别数量:", train_df["target"].nunique(dropna=False))
    print("缺失标签数:", int(train_df["target"].isna().sum()))

    # 5. 检查重复数据
    print("\n========== 重复数据检查 ==========")
    print("训练集重复行数:", int(train_df.duplicated().sum()))
    print("训练集重复文本数:", int(train_df["text"].duplicated().sum()))
    print("测试集重复文本数:", int(test_df["text"].duplicated().sum()))

    # 6. 统计文本长度
    print("\n========== 文本统计 ==========")
    print("训练集:", text_statistics(train_df["text"]))
    print("测试集:", text_statistics(test_df["text"]))

    # 7. 检查预测模板，避免覆盖原文件
    print("\n========== predictions.csv ==========")
    if pred_path.exists():
        with pred_path.open(
            "r", encoding="utf-8-sig", errors="replace"
        ) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(repr(line.rstrip("\r\n")))
    else:
        print("预测模板文件不存在。")


if __name__ == "__main__":
    main()