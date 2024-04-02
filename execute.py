# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from itertools import combinations


def main():
    # 数据文件路径，根据实际情况修改
    data_path = 'data/words_data.xlsx'
    # 模型路径,根据实际情况修改
    model_path = "./models/tencent-ailab-embedding.txt"
    # 结果保存路径，根据实际情况修改
    data_save_path = './data/DAT_scores.xlsx'

    print('加载模型中...')
    try:
        # 加载模型，请注意！加载时间可能较长，取决于电脑性能、模型大小，可使用limit参数限制加载的词向量数量
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, encoding="utf8",
                                                  unicode_errors='ignore')
    except Exception as e:
        print(f"加载模型失败！{e}")
        exit(1)
    print('加载完毕')

    # 读取数据
    df = pd.read_excel(data_path)
    # 计算DAT得分,并写入到数据中
    result = df.apply(lambda row: calculate_score(row, model), axis=1, result_type='expand')
    df['DAT得分'], df['替换信息'] = result[0], result[1]
    # 保存结果
    df.to_excel(data_save_path, index=False)


def calculate_score(row, model):
    # 获取10个词
    words = [row[f"词{i}"] for i in range(1, 11)]
    # 检查词是否在模型中
    valid_words, replacements = get_valid_words(words, model)
    if valid_words == 0:
        print(f"序号：{row['序号']} 有效词异常，无法计算。")
        return 0
    if replacements:
        print(f"替换信息：序号：{row['序号']} {replacements}")
    # 生成所有可能的词对
    word_pairs = list(combinations(valid_words, 2))
    # 计算所有词对的余弦距离
    distances = [cosine_distance(word1, word2, model) for word1, word2 in word_pairs]
    # 计算DAT得分
    dat_score = np.mean(distances)
    return dat_score, "; ".join([f"{orig}->{rep}" for _, orig, rep in replacements])


def get_valid_words(row, model):
    replacements = []  # 用于记录替换的信息
    valid_words = []  # 已验证词的列表
    used_replacements = []  # 已使用的替换词列表

    # 先处理前7个词
    for i in range(7):
        word = row[i]
        if word in model.key_to_index:
            valid_words.append(word)
        else:
            # 如果当前词不存在，则尝试使用后续的词替换
            replaced = False
            for replacement in row[7:]:
                if replacement in model.key_to_index and replacement not in used_replacements:
                    valid_words.append(replacement)
                    replacements.append((i + 1, word, replacement))  # 记录替换信息
                    used_replacements.append(replacement)  # 记录已使用的替换词
                    replaced = True
                    break
            if not replaced:  # 如果找不到替换词
                print(f"无法找到替换词来代替'{word}'。")
                return 0, []  # 表示这一行处理失败

    if len(valid_words) < 7:  # 如果有效词数量不足
        print("有效词数量不足。")
        return 0, []

    # 返回有效词列表和替换信息
    return valid_words, replacements


def cosine_distance(word1, word2, model):
    # 获取词向量
    vector1 = model[word1]
    vector2 = model[word2]
    # 计算余弦相似度
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    # 将余弦相似度转换为距离，并乘以100
    result = (1 - cosine_similarity) * 100
    return result


if __name__ == '__main__':
    main()
