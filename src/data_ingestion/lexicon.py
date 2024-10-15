import requests
import pandas as pd
from io import StringIO
from nltk.tokenize import word_tokenize

def load_lexicon(lexicon_url):
    try:
        content = requests.get(lexicon_url).text
        df_lexicon = pd.read_csv(StringIO(content), sep='\t', header=None, skiprows=1)
        return dict(zip(df_lexicon[0], df_lexicon[1]))
    except Exception as e:
        print(f"Failed to load lexicon: {e}")
        return {}

def calculate_sentiment_score(text, lexicon_positive, lexicon_negative):
    """Calculate sentiment base of polarity score
    
    Args:
        text (str): review text
        lexicon_positive (str): link of lexicon_positive data
    """
    tokens = word_tokenize(text)
    score = sum(lexicon_positive.get(word, 0) - lexicon_negative.get(word, 0) for word in tokens)
    
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    
    return score, polarity
