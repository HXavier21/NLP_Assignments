from nltk.util import ngrams
from collections import Counter

# 定义输入文件名和输出文件名
input_file_name = 'sentences.txt'
output_file_name = 'ngram_probabilities.txt'

# 读取文件中的句子
with open(input_file_name, 'r', encoding='utf-8') as infile:
    sentences = infile.readlines()

# 将每个句子合并为一个大字符串
corpus = ' '.join(sentences)

# 分词
words = corpus.split()

# 生成各阶n-gram
uni_grams = list(ngrams(words, 1))
bi_grams = list(ngrams(words, 2))
tri_grams = list(ngrams(words, 3))

# 计算各阶n-gram频率
uni_gram_freq = Counter(uni_grams)
bi_gram_freq = Counter(bi_grams)
tri_gram_freq = Counter(tri_grams)

# 计算各阶n-gram概率
total_unigrams = sum(uni_gram_freq.values())
uni_gram_prob = {uni_gram: freq / total_unigrams for uni_gram, freq in uni_gram_freq.items()}

bi_gram_prob = {}
for bi_gram, freq in bi_gram_freq.items():
    prefix = bi_gram[0]
    bi_gram_prob[bi_gram] = freq / uni_gram_freq[(prefix,)]

tri_gram_prob = {}
for tri_gram, freq in tri_gram_freq.items():
    prefix = tri_gram[:2]
    tri_gram_prob[tri_gram] = freq / bi_gram_freq[prefix]

# 保存结果到文件
with open(output_file_name, 'w', encoding='utf-8') as outfile:
    outfile.write("Unigram Probabilities:\n")
    for uni_gram, prob in uni_gram_prob.items():
        outfile.write(f"{uni_gram}\t{prob}\n")

    outfile.write("\nBigram Probabilities:\n")
    for bi_gram, prob in bi_gram_prob.items():
        outfile.write(f"{bi_gram}\t{prob}\n")

    outfile.write("\nTrigram Probabilities:\n")
    for tri_gram, prob in tri_gram_prob.items():
        outfile.write(f"{tri_gram}\t{prob}\n")

print(f"N-gram probabilities have been saved to '{output_file_name}'.")
