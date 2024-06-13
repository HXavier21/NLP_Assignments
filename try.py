from language_model import build_language_model
import random

def sample_trigram_frequencies(trigram_freq, sample_size=5):
    sampled_trigrams = random.sample(list(trigram_freq.items()), k=sample_size)
    print("Sampled Trigram Frequencies:")
    for trigram, freq in sampled_trigrams:
        print(f"{trigram}: {freq}")

def sample_bigram_frequencies(bigram_freq, sample_size=5):
    sampled_bigrams = random.sample(list(bigram_freq.items()), k=sample_size)
    print("Sampled Bigram Frequencies:")
    for bigram, freq in sampled_bigrams:
        print(f"{bigram}: {freq}")

trigram_freq, bigram_freq, unigram_freq = build_language_model("ngram_frequencies.txt")

# 随机抽查几个三元组
sample_trigram_frequencies(trigram_freq)

sample_bigram_frequencies(bigram_freq)