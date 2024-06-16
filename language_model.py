from collections import defaultdict
def build_language_model(file_name):
    trigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)
    unigram_freq = defaultdict(int)

    with open(file_name, 'r', encoding='utf-8') as infile:
        current_section = None
        for line in infile:
            line = line.strip()
            if line == "Unigram Frequencies:":
                current_section = "unigram"
            elif line == "Bigram Frequencies:":
                current_section = "bigram"
            elif line == "Trigram Frequencies:":
                current_section = "trigram"
            elif line:
                ngram, freq = line.rsplit('\t', 1)
                ngram = tuple(eval(ngram))
                freq = int(freq)
                if current_section == "unigram":
                    unigram_freq[ngram] = freq
                elif current_section == "bigram":
                    bigram_freq[ngram] = freq
                elif current_section == "trigram":
                    trigram_freq[ngram] = freq

    return trigram_freq, bigram_freq, unigram_freq