import collections
import nltk
from nltk.corpus import reuters
from nltk.util import ngrams
from tqdm import tqdm
from language_model import build_language_model


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


def exchange_letters(word):
    return_set = []
    word_list = list(word)

    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            # 创建一个新单词，交换第i和j个字母
            new_word_list = word_list[:]
            new_word_list[i], new_word_list[j] = new_word_list[j], new_word_list[i]
            return_set.append(''.join(new_word_list))
    return return_set


# 信道模型：常见编辑操作
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    exchange = exchange_letters(word)
    return_set = set(deletes + replaces + inserts + exchange)
    return return_set


def edits1_with_upper_letter(word):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    exchange = exchange_letters(word)
    return_set = set(deletes + replaces + inserts + exchange)
    return return_set


def build_reuters_language_model():
    words = reuters.words()
    trigrams = ngrams(words, 3)
    bigrams = ngrams(words, 2)
    unigrams = words

    trigram_freq = collections.Counter(trigrams)
    bigram_freq = collections.Counter(bigrams)
    unigram_freq = collections.Counter(unigrams)

    return trigram_freq, bigram_freq, unigram_freq


def trigram_last_probability(trigram_freq, bigram_freq, w1, w2, w3):
    w1_w2_freq = bigram_freq[(w1, w2)]
    result = trigram_freq[(w1, w2, w3)] / w1_w2_freq if w1_w2_freq > 0 else 0
    if result == 0:
        result = bigram_probability(bigram_freq, w2, w3)
    return result


def trigram_middle_probability(bigram_freq, w1, w2, w3):
    w1_w2_freq = bigram_probability(bigram_freq, w1, w2)
    w2_w3_freq = bigram_probability(bigram_freq, w2, w3)
    # result = w1_w2_freq * w2_w3_freq
    # if result == 0:
    result = max(w1_w2_freq, w2_w3_freq)
    return result


def trigram_first_probability(trigram_freq, bigram_freq, w1, w2, w3):
    w2_w3_freq = bigram_freq[(w2, w3)]
    result = trigram_freq[(w1, w2, w3)] / w2_w3_freq if w2_w3_freq > 0 else 0
    if result == 0:
        result = bigram_probability(bigram_freq, w1, w2)
    return result


def max_trigram_probability(trigram_freq, bigram_freq, w1, w2, w, w4, w5):
    return max(trigram_first_probability(trigram_freq, bigram_freq, w, w4, w5),
               trigram_middle_probability(bigram_freq, w2, w, w4),
               trigram_last_probability(trigram_freq, bigram_freq, w1, w2, w))


def bigram_probability(bigram_freq, w1, w2):
    # w1_freq = sum(freq for (first_word, _), freq in bigram_freq.items() if first_word == w1)
    # w2_freq = sum(freq for (_, second_word), freq in bigram_freq.items() if second_word == w2)
    # w1_w2_freq = bigram_freq[(w1, w2)]
    # return w1_w2_freq / (w1_freq + w2_freq - w1_w2_freq) if w1_freq > 0 and w2_freq > 0 else 0
    w1_w2_freq = bigram_freq[(w1, w2)]
    whole_freq = 0
    for (first_word, second_word), freq in bigram_freq.items():
        if first_word == w1 or second_word == w2:
            whole_freq += 1
    return w1_w2_freq / whole_freq if whole_freq > 0 else 0


def unigram_probability(unigram_freq, word):
    total_words = sum(unigram_freq.values())
    return unigram_freq[word] / total_words if total_words > 0 else 0


def generate_upper_candidates(word, vocab):
    upper_candidate_list = edits1_with_upper_letter(word)
    checked_upper_candidates = check(upper_candidate_list, vocab)
    if checked_upper_candidates:
        return checked_upper_candidates
    else:
        further_upper_candidates = []
        for candidate in upper_candidate_list:
            further_edits = edits1_with_upper_letter(candidate)
            further_upper_candidates.extend(further_edits)
        checked_further_upper_candidates = check(further_upper_candidates, vocab)
        print(checked_further_upper_candidates)
        if checked_further_upper_candidates:
            return checked_further_upper_candidates
        # 如果两轮都没有找到，返回空列表
        print('cannot edit: ' + word)
        return []


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
        else:
            return generate_upper_candidates(word, vocab)


def check_if_skip(word, check_set):
    if word in check_set or check_if_special_alpha(word):
        return True
    else:
        return False


def check_if_special_alpha(word):
    if not word.isalpha() or word == "'s":
        return True
    else:
        return False


def transform_special_alpha(word):
    if check_if_special_alpha(word):
        if word == "'s":
            return 'is'
        elif word.isnumeric():
            return 'one'
        else:
            return word
    else:
        return word


def get_best_candidate(candidate_list, corrected_sentence, i):
    if candidate_list:
        if i == 0:
            best_candidate = max(candidate_list,
                                 key=lambda w: trigram_first_probability(trigram_freq, bigram_freq,
                                                                         w,
                                                                         transform_special_alpha(
                                                                             corrected_sentence[1]),
                                                                         transform_special_alpha(
                                                                             corrected_sentence[2])))
        elif i < len(corrected_sentence) - 2:
            best_candidate = max(candidate_list,
                                 key=lambda w: trigram_middle_probability(bigram_freq,
                                                                          transform_special_alpha(
                                                                              corrected_sentence[i - 1]),
                                                                          w,
                                                                          transform_special_alpha(
                                                                              corrected_sentence[i + 1])))
        else:
            if i <= 1:
                best_candidate = max(candidate_list,
                                     key=lambda w: trigram_middle_probability(bigram_freq,
                                                                              transform_special_alpha(
                                                                                  corrected_sentence[i - 1]),
                                                                              w,
                                                                              transform_special_alpha(
                                                                                  corrected_sentence[i + 1])))
            else:
                previous_two_words = corrected_sentence[i - 2:i]
                best_candidate = max(candidate_list, key=lambda w: trigram_last_probability(trigram_freq, bigram_freq,
                                                                                            transform_special_alpha(
                                                                                                previous_two_words[0]),
                                                                                            transform_special_alpha(
                                                                                                previous_two_words[1]),
                                                                                            w))
        return best_candidate
    else:
        return corrected_sentence[i]


# 这里假设 edits1 和 check 函数已经被定义
# edits1 函数用于生成给定单词的编辑候选词
# check 函数用于检查给定的候选词列表中哪些词在词汇表中


def correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab):
    words = nltk.word_tokenize(sentence)  # 使用NLTK的word_tokenize分词，可以处理标点符号
    corrected_sentence = [item for item in words]
    non_word_errors = set()
    count = 0

    # First pass: correct non-word errors
    for i, word in enumerate(words):
        if check_if_skip(word, vocab):  # 保留标点符号或非字母字符
            continue
        else:
            count += 1
            non_word_errors.add(word)
            candidate_list = generate_candidates(word, vocab)

            corrected_sentence[i] = get_best_candidate(candidate_list, corrected_sentence, i)

    real_word_count = error_count - count
    # print(corrected_sentence + [real_word_count])
    if real_word_count > 0:
        prob = []
        for i, word in enumerate(corrected_sentence):
            if check_if_skip(word, non_word_errors):
                continue  # Skip words that were already corrected as non-word errors
            probability = unigram_probability(unigram_freq, word)
            # if i == 0:
            #     probability = unigram_probability(unigram_freq, word)
            # elif i == 1:
            #     previous_word = corrected_sentence[0]
            #     probability = bigram_probability(bigram_freq, previous_word, word)
            # else:
            #     previous_two_words = corrected_sentence[i - 2:i]
            #     probability = trigram_probability(trigram_freq, bigram_freq, previous_two_words[0],
            #                                       previous_two_words[1], word)
            prob.append((probability, i, word))

        prob.sort()
        # print(prob)
        # Correct the real-word errors with the lowest probabilities
        for _, i, word in prob:
            if real_word_count == 0:
                break
            candidate_list = generate_candidates(word, vocab)
            best_candidate = get_best_candidate(candidate_list, corrected_sentence, i)

            if corrected_sentence[i] == best_candidate:
                continue
            else:
                corrected_sentence[i] = best_candidate
                real_word_count = real_word_count - 1

    return ' '.join(corrected_sentence)


# 仅将结果写入文件
def correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path, pbar):
    with open(output_path, 'w') as output_file:
        for sentence_id, error_count, sentence in sentences:
            corrected_sentence = correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab)
            output_file.write(f"{sentence_id}\t{corrected_sentence}\n")
            pbar.update(1)


# 加载数据和词汇表，构建模型
vocab_path = 'vocab.txt'
vocab = load_vocab(vocab_path)
#trigram_freq, bigram_freq, unigram_freq = build_reuters_language_model()
trigram_freq, bigram_freq, unigram_freq = build_language_model('combined_ngram_frequencies.txt')
file_path = input('testData:')
output_path = 'result.txt'
sentences = load_data(file_path)

with tqdm(total=1000) as pbar:
    correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path, pbar)
print(trigram_middle_probability(bigram_freq, 'amount', 'dent', 'in'))
print(trigram_middle_probability(bigram_freq, 'amount', 'spent', 'in'))
print(trigram_middle_probability(bigram_freq, 'amount', 'sent', 'in'))
