import pandas as pd
from scraper import gather_reviews
from preprocessing import preprocess_text
from lexicon import load_lexicon, calculate_sentiment_score

# Scrape reviews
url_list = [
    'https://tokopedia.link/egeGBRvz5Kb',
    'https://tokopedia.link/efLq5V3y5Kb',
    # Add more URLs as needed
]
df_reviews = gather_reviews(url_list)

# Preprocess text
df_reviews['cleaned_reviews'] = df_reviews['review'].apply(preprocess_text)
df_reviews.to_csv('cleaned_reviews.csv', index=False)

# Load lexicon
lexicon_positive = load_lexicon('https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv')
lexicon_negative = load_lexicon('https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv')

# Calculate sentiment score
df_reviews[['sentiment_score', 'sentiment_label']] = df_reviews['cleaned_reviews'].apply(
    lambda x: calculate_sentiment_score(x, lexicon_positive, lexicon_negative)
).apply(pd.Series)

# Save labeled reviews
df_reviews.to_csv('labeled_reviews_with_lexicon.csv', index=False)