import nltk
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize

# 下载路透社语料库和 punkt 句子分割模型
nltk.download('reuters')
nltk.download('punkt')

# 获取所有文件ID
file_ids = reuters.fileids()

# 初始化句子计数
sentence_count = 0

# 打开一个文件来写句子
with open('reuters_sentences.txt', 'w', encoding='utf-8') as outfile:
    # 遍历所有文件ID
    for file_id in file_ids:
        # 获取文件的原始文本
        text = reuters.raw(file_id)
        # 使用 NLTK 分割成句子
        sentences = sent_tokenize(text)
        # 更新句子计数
        sentence_count += len(sentences)
        # 写入每个句子到文件，每个句子占一行
        for sentence in sentences:
            outfile.write(sentence + '\n')

# 打印总句子数
print(f'Total number of sentences: {sentence_count}')
