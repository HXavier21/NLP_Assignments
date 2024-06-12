import torch
from transformers import BertTokenizer, BertForMaskedLM
import nltk
import numpy as np
import difflib, math
import Levenshtein
from tqdm import tqdm

nltk.download('punkt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='models')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')  # 用BertForMaskedLM类加载模型，该类可以对句子中的标记字符[MASK]进行预测。
model.eval()
model.to(device)

vocab = open('vocab.txt', 'r')
vocab_dict = {}
vocab_arr = []
for word in vocab.readlines():
    word = word.strip()
    vocab_dict[word] = len(vocab_arr)
    vocab_arr.append(word)

# 2000
K = 2000


def correct_topk(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_index = indexed_tokens.index(103)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([[0 for _ in range(len(indexed_tokens))]]).to(device)
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]
    predicted = torch.topk(predictions[0, masked_index], k=K)
    predicted_indexs = predicted.indices
    predicted_values = predicted.values
    predicted_tokens = [tokenizer.convert_ids_to_tokens([predicted_index])[0] for predicted_index in
                        predicted_indexs.tolist()]
    return predicted_tokens, predicted_values.tolist()


def string_sim(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def edit_sim(s1, s2):  # 83.8
    return 1.0 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2))


SUFFIXS = ['s', 'es', 'ly']
SPEC_NONES = ['january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november',
              'december']

# 0.018
R = 0.018


def select(word, vocab, prio):
    score = [edit_sim(vocab_arr[i], word) for i in range(len(vocab_arr))]
    for i in range(len(vocab)):
        v = vocab[i]
        if vocab_dict.__contains__(v):
            score[vocab_dict[v]] += R * prio[i]
    return vocab_arr[torch.tensor(score).argmax().item()]


input = open('testdata.txt', 'r')
output = open('result.txt', 'w')

# 剩下的任务：
# 1. 单词词性纠正
# 2. 单词大小写纠正（专有名词和首字母等）
# 3. 单词时态纠正
# 4. 单词单复数纠正

for i in tqdm(range(1000)):
    parts = input.readline().split('\t')
    cnt = int(parts[1])
    line = parts[2]

    words = nltk.word_tokenize(line)
    for j in range(len(words)):
        if vocab_dict.__contains__(words[j]):
            continue
        origin = words[j]
        words[j] = '[MASK]'
        top_words, top_values = correct_topk(f"[CLS] {' '.join(words)} [SEP]")
        words[j] = select(origin, top_words, top_values)
        if j == 0 and words[j].islower():
            words[j] = words[j].capitalize()
        cnt -= 1
        for suffix in SUFFIXS:
            if origin[-1] == suffix and vocab_dict.__contains__(words[j] + suffix):
                words[j] += suffix
                break
        if origin == origin.capitalize() and vocab_dict.__contains__(words[j].capitalize()):
            words[j] = words[j].capitalize()
        elif origin.isupper() and vocab_dict.__contains__(words[j].upper()):
            words[j] = words[j].upper()

    print(f"Task {i} left {cnt} words.")
    while cnt > 0:
        # 词性纠正、时态纠正等
        pass
        cnt -= 1

    output.write(str(i) + '\t' + ' '.join(words) + '\n')
    output.flush()
