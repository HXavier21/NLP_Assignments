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

# 信道模型：常见编辑操作
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    # Move first letter to the end
    move_first_to_end = [word[1:] + word[0]]
    # Move last letter to the beginning
    move_last_to_beginning = [word[-1] + word[:-1]]
    # Change case
    change_case = [word.swapcase()]
    # Swap adjacent letters
    swaps = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    return set(deletes + transposes + replaces + inserts + move_first_to_end + move_last_to_beginning + change_case + swaps)

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
    # 生成第一轮编辑后的候选词
    candidate_list = edits1(word)
    
    # 将词汇表转换为集合，以提高查找效率
    vocab_set = set(vocab)
    
    # 在第一轮编辑后的候选词中寻找在词汇表中存在的候选词
    checked_candidates = [candidate for candidate in candidate_list if candidate in vocab_set]
    
    # 如果在词汇表中找到了候选词，直接返回这些候选词
    if checked_candidates:
        return checked_candidates
    
    # 生成第二轮编辑后的候选词
    further_candidates = set()
    for candidate in candidate_list:
        further_candidates.update(edits1(candidate))
    
    # 在第二轮编辑后的候选词中寻找在词汇表中存在的候选词
    checked_further_candidates = [candidate for candidate in further_candidates if candidate in vocab_set]
    
    # 如果在词汇表中找到了候选词，返回这些候选词
    if checked_further_candidates:
        return checked_further_candidates
    
    # 如果两轮都没有找到，返回空列表
    return []

def correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab):
    words = nltk.word_tokenize(sentence)  # 使用NLTK的word_tokenize分词，可以处理标点符号
    corrected_sentence = []
    non_word_errors = set()
    count = 0

    # First pass: correct non-word errors
    for i, word in enumerate(words):
        if word in vocab or not word.isalpha():  # 保留标点符号或非字母字符
            corrected_sentence.append(word)
        else:
            count += 1
            non_word_errors.add(word)
            candidate_list = generate_candidates(word, vocab)
            
            if candidate_list:
                if i > 3:
                    previous_context = corrected_sentence[i-3:i]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq, previous_context[-2], previous_context[-1], w) * bigram_probability(bigram_freq, previous_context[-1], w))
                elif i > 1:
                    previous_context = corrected_sentence[:i]
                    best_candidate = max(candidate_list, key=lambda w: bigram_probability(bigram_freq, previous_context[-1], w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                corrected_sentence.append(best_candidate)
            else:
                corrected_sentence.append(word)

    real_word_count = error_count - count
    if real_word_count > 0:
        prob = []
        for i, word in enumerate(words):
            if word in non_word_errors or not word.isalpha():
                continue  # Skip words that were already corrected as non-word errors
            if i > 3:
                previous_context = corrected_sentence[i-3:i]
                probability = trigram_probability(trigram_freq, bigram_freq, previous_context[-2], previous_context[-1], word)
            elif i > 1:
                previous_context = corrected_sentence[:i]
                probability = bigram_probability(bigram_freq, previous_context[-1], word)
            else:
                probability = unigram_probability(unigram_freq, word)
            prob.append((probability, i, word))

        prob.sort()

        # Correct the real-word errors with the lowest probabilities
        for _, i, word in prob[:real_word_count]:
            candidate_list = generate_candidates(word, vocab)
            if candidate_list:
                if i > 3:
                    previous_context = corrected_sentence[i-3:i]
                    best_candidate = max(candidate_list, key=lambda w: trigram_probability(trigram_freq, bigram_freq, previous_context[-2], previous_context[-1], w) * bigram_probability(bigram_freq, previous_context[-1], w))
                elif i > 1:
                    previous_context = corrected_sentence[:i]
                    best_candidate = max(candidate_list, key=lambda w: bigram_probability(bigram_freq, previous_context[-1], w))
                else:
                    best_candidate = max(candidate_list, key=lambda w: unigram_probability(unigram_freq, w))
                corrected_sentence[i] = best_candidate

    return ' '.join(corrected_sentence)

def correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path):
    with open(output_path, 'w') as output_file:
        for sentence_id, error_count, sentence in sentences:
            corrected_sentence = correct_sentence(sentence, error_count, trigram_freq, bigram_freq, unigram_freq, vocab)
            output_file.write(f"{sentence_id}\t{corrected_sentence}\n")


# 加载数据和词汇表，构建模型
vocab_path = 'vocab.txt'
file_path = 'realwordtestdata.txt'
output_path = 'realwordresult.txt'
vocab = load_vocab(vocab_path)
sentences = load_data(file_path)
trigram_freq, bigram_freq, unigram_freq = build_language_model()

# 纠正并写入文件
correct_and_save(sentences, trigram_freq, bigram_freq, unigram_freq, vocab, output_path)

