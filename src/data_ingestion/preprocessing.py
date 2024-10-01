import string
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import replace_slang, replace_word_elongation, remove_stopwords

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def remove_punctuation(word_list):
    punctuation_list = string.punctuation
    return [word for word in word_list if word not in punctuation_list]

def remove_numbers(word_list):
    return [word for word in word_list if word.isalpha()]

def stemming_words(word_list):
    return [stemmer.stem(word) for word in word_list]

def remove_emoticons(text):
    emoji_pattern = re.compile(
        "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U00002702-\U000027B0"  # other symbols
            u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_text(text):
    text = text.lower()
    text = replace_word_elongation(text)
    text = replace_slang(text)
    text = remove_stopwords(text)
    text = remove_emoticons(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([!?.,"])\1+', r'\1', text)
    
    words = word_tokenize(text)
    words = remove_punctuation(words)
    words = remove_numbers(words)
    words = stemming_words(words)
    return ' '.join(words)


