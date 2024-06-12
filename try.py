import re
import collections
import nltk
from nltk.corpus import reuters
from nltk.util import ngrams

# 下载所需的nltk资源
nltk.download('reuters')
nltk.download('punkt')


# 加载词汇表
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = set(file.read().split())
    return vocab


# 数据预处理
def load_data(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentence_id, error_count, sentence = line.strip().split('\t')
            sentences.append((sentence_id, int(error_count), sentence))
    return sentences


# 检查候选单词是否在词汇表中
def check(candidate_list, vocab):
    return [word for word in candidate_list if word in vocab]


# 信道模型：常见编辑操作
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return_set = set(deletes + transposes + replaces + inserts)
    for letter in word:
        if letter.isupper():
            return [item[0].upper() + item[1:] for item in return_set]
    return return_set


def build_language_model():
    words = reuters.words()
    trigrams = ngrams(words, 3)
    bigrams = ngrams(words, 2)
    unigrams = words

    trigram_freq = collections.Counter(trigrams)
    bigram_freq = collections.Counter(bigrams)
    unigram_freq = collections.Counter(unigrams)

    return trigram_freq, bigram_freq, unigram_freq


def trigram_probability(trigram_freq, bigram_freq, w1, w2, w3):
    w1_w2_freq = bigram_freq[(w1, w2)]
    return trigram_freq[(w1, w2, w3)] / w1_w2_freq if w1_w2_freq > 0 else 0


def bigram_probability(bigram_freq, w1, w2):
    w1_freq = sum(freq for (first_word, _), freq in bigram_freq.items() if first_word == w1)
    return bigram_freq[(w1, w2)] / w1_freq if w1_freq > 0 else 0


def unigram_probability(unigram_freq, word):
    total_words = sum(unigram_freq.values())
    return unigram_freq[word] / total_words if total_words > 0 else 0


def generate_candidates(word, vocab):
    # 首先生成第一轮编辑后的候选词
    candidate_list = edits1(word)

    # 检查这些候选词是否在词汇表中
    checked_candidates = check(candidate_list, vocab)

    # 如果在词汇表中找到了候选词，直接返回这些候选词
    if checked_candidates:
        return checked_candidates

    # 如果第一轮没有找到，进行第二轮编辑操作
    else:
        # 对第一轮的每个候选词进行进一步编辑
        further_candidates = []
        for candidate in candidate_list:
            further_edits = edits1(candidate)
            further_candidates.extend(further_edits)

        # 检查第二轮编辑后的候选词是否在词汇表中
        checked_further_candidates = check(further_candidates, vocab)

        # 如果在词汇表中找到了候选词，返回这些候选词
        if checked_further_candidates:
            return checked_further_candidates
        # 如果两轮都没有找到，返回空列表
        print('cannot edit: ' + word)
        return []


def check_if_skip(word, check_set):
    if word in check_set or not word.isalpha() or word == "'s":
        return True
    else:
        return False


# 这里假设 edits1 和 check 函数已经被定义
# edits1 函数用于生成给定单词的编辑候选词
# check 函数用于检查给定的候选词列表中哪些词在词汇表中


def correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab):
    words = nltk.word_tokenize(sentence)  # 使用NLTK的word_tokenize分词，可以处理标点符号
    corrected_sentence = []
    non_word_errors = set()
    count = 0

    # First pass: correct non-word errors
    for i, word in enumerate(words):
        if check_if_skip(word, vocab):  # 保留标点符号或非字母字符
            corrected_sentence.append(word)
        else:
            count += 1
            non_word_errors.add(word)
            candidate_list = generate_candidates(word, vocab)

            if candidate_list:
                if i > 1:
                    previous_two_words = corrected_sentence[-2:]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq,
                                                                                           previous_two_words[0],
                                                                                           previous_two_words[1], w))
                elif i == 1:
                    previous_word = corrected_sentence[-1]
                    best_candidate = max(candidate_list,
                                         key=lambda w: bigram_probability(bigram_freq, previous_word, w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                corrected_sentence.append(best_candidate)
            else:
                corrected_sentence.append(word)

    real_word_count = error_count - count
    for i, word in enumerate(corrected_sentence):
        if word == "'s":
            corrected_sentence[i] = 'is'
    print(corrected_sentence + [real_word_count])
    if real_word_count > 0:
        prob = []
        for i, word in enumerate(corrected_sentence):
            if check_if_skip(word, non_word_errors):
                continue  # Skip words that were already corrected as non-word errors
            if i == 0:
                probability = unigram_probability(unigram_freq, word)
            elif i == 1:
                previous_word = corrected_sentence[0]
                probability = bigram_probability(bigram_freq, previous_word, word)
            else:
                previous_two_words = corrected_sentence[i - 2:i]
                probability = trigram_probability(trigram_freq, bigram_freq, previous_two_words[0],
                                                  previous_two_words[1], word)
            prob.append((probability, i, word))

        prob.sort()
        print(prob)
        # Correct the real-word errors with the lowest probabilities
        for _, i, word in prob[:real_word_count]:
            candidate_list = generate_candidates(word, vocab)
            if candidate_list:
                if i > 1:
                    previous_two_words = corrected_sentence[i - 2:i]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq,
                                                                                           previous_two_words[0],
                                                                                           previous_two_words[1], w))
                elif i == 1:
                    previous_word = corrected_sentence[0]
                    best_candidate = max(candidate_list,
                                         key=lambda w: bigram_probability(bigram_freq, previous_word, w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                corrected_sentence[i] = best_candidate

    return ' '.join(corrected_sentence)


# 仅将结果写入文件
def correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path):
    with open(output_path, 'w') as output_file:
        for sentence_id, error_count, sentence in sentences:
            corrected_sentence = correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab)
            output_file.write(f"{sentence_id}\t{corrected_sentence}\n")


# 加载数据和词汇表，构建模型
vocab_path = 'vocab.txt'
file_path = input('testData:')
output_path = 'result.txt'
vocab = load_vocab(vocab_path)
sentences = load_data(file_path)
trigram_freq, bigram_freq, unigram_freq = build_language_model()

# 纠正并写入文件
#correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path)
print(trigram_probability(trigram_freq, bigram_freq, 'what', 'is', 'interesting'))
# print(generate_candidates("ther", vocab))
