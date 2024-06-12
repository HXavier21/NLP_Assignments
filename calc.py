# encoding=utf-8
import math
import numpy as np
import difflib
from spelling_correcter import correct_text_generic

input = open('testdata.txt', 'r')
output = open('result.txt', 'w')


def lcs_sim(s, t):  # 27.7
    len1 = len(s)
    len2 = len(t)
    # 初始化一个二维数组，行数为t的大小，列数为s的大小
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if t[i - 1] == s[j - 1]:
                res[i][j] = 1 + res[i - 1][j - 1]
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    return res[-1][-1] / math.sqrt(len1 * len2)


def eculid_sim(x, y):
    """
    欧几里得相似度
    """
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def string_sim(s1, s2):  # 24.7
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


vocab = open('vocab.txt', 'r')
vocab_dict = {}
vocab_arr = []
for word in vocab.readlines():
    word = word[:-1]
    vocab_dict[word] = 1
    vocab_arr.append(word)


def correct_text(word):
    res = vocab_arr[0]
    sim = 0
    for vocab in vocab_arr:
        sim2 = lcs_sim(vocab, word)
        if sim2 > sim:
            sim = sim2
            res = vocab
    return res


cnt = 0
for line in input.readlines():
    cnt += 1
    line = line.replace('\t', ' ').replace('\n', ' ')
    words = line.split(' ')
    err = int(words[1])
    words = words[2:]
    ans = []
    for word in words:
        if len(word) == 0:
            continue
        if vocab_dict.__contains__(word):
            ans.append(word)
        else:
            for i in range(len(word) - 1, 0, -1):
                if vocab_dict.__contains__(word[:i]) and vocab_dict.__contains__(word[i:]):
                    ans.append(word)
                    break
                elif vocab_dict.__contains__(word[i:]):
                    ans.append(correct_text_generic(word[:i]) + word[i:])
                    break
            else:
                ans.append(correct_text_generic(word))
    output.write(str(cnt) + '\t' + ' '.join(ans) + '\n')
    output.flush()
