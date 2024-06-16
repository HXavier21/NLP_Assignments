# If you want to run the demo in your local environment
# Please pip install the package
# And run the following command:
# streamlit run webui.py

import streamlit as st
from annotated_text import annotated_text
from nltk.tokenize import word_tokenize

from spelling_corrector import load_vocab, build_reuters_language_model, correct_sentence


# 需要加一下缓存 不然每次交互重新加载很耗时
@st.cache_data
def load_model_and_data():
    vocab_path = 'vocab.txt'
    vocab = load_vocab(vocab_path)
    trigram_freq, bigram_freq, unigram_freq = build_reuters_language_model()
    return vocab, trigram_freq, bigram_freq, unigram_freq


def main():
    st.subheader("Spelling Checker", divider='rainbow')

    vocab, trigram_freq, bigram_freq, unigram_freq = load_model_and_data()

    text_input = st.text_area('Input your text', height=100)
    error_count = st.number_input('Error count', min_value=0, max_value=10, value=0)

    if st.button("Correct"):
        st.divider()
        if text_input and error_count >= 0:
            corrected_text = correct_sentence(text_input, error_count, trigram_freq, bigram_freq, unigram_freq, vocab)
            st.write("Correct result:")
            show_differences(text_input, corrected_text)
        else:
            st.error("Please input valid information")


# 标注修改过的单词 用的Annotated Text Component for Streamlit
def show_differences(original_text, corrected_text):
    original_words = word_tokenize(original_text)
    corrected_words = word_tokenize(corrected_text)

    annotated_result = []
    for original, corrected in zip(original_words, corrected_words):
        if original.lower() != corrected.lower():
            annotated_result.append((corrected, "", "#faa"))
        else:
            annotated_result.append(corrected)
        annotated_result.append(" ")

    annotated_text(*annotated_result)


if __name__ == "__main__":
    main()
