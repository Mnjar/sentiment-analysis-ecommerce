import json
import tensorflow as tf
from transformers import XLMRobertaTokenizer
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load hyperparameters
with open('config.json') as f:
    config = json.load(f)
    
def rnns_tokenizer(data, max_words=5000, max_len=250):
    """Tokenize data for rnns model

    Args:
        data (Data): Dataframe
        max_words (int, optional): max word. Defaults to 5000.
        max_len (int, optional): max len. Defaults to 250.

    Returns:
        sequence: string sequence
        tokenize: string tokenize
    """
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(data['cleaned_reviews'].values)
    sequences = tokenizer.texts_to_sequences(data['cleaned_reviews'].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, tokenizer

def load_roberta_tokenizer(tokenizer_name):
    return XLMRobertaTokenizer.from_pretrained(tokenizer_name)

def tokenize_data(tokenizer, max_length, reviews=None, data=None, is_inference: bool=False):
    """ Tokenizes the given text data using the provided tokenizer.

    Args:
        `data (Dataframe)`: DataFrame containing at least the 'cleaned_reviews' and 'sentiment_label' columns.
        `tokenizer (PreTrainedTokenizer`): A tokenizer instance used to tokenize the text.
        `max_length (int)`: The maximum length of tokens to which the text should be truncated or padded.
        `reviews (list of str, optional)`: A list of reviews to be tokenized when provided. If None, tokenization will be performed
        on the 'cleaned_reviews' column of the `data` DataFrame.. Defaults to None.
        `is_inference (bool, optional)`:  If True, tokenization is done on the provided `reviews` (for inference use cases).
        Default is False (for training use cases).

    Returns:
        input_ids: tf.Tensor
        A tensor of token IDs for the tokenized text data.
    attention_masks: tf.Tensor
        A tensor of attention masks indicating which tokens are padding.
    labels: tf.Tensor
        One-hot encoded labels for sentiment classification.
        
    Notes:
    ------
    - For inference, pass the `reviews` parameter to tokenize custom input.
    - For training, ensure that `data` contains the necessary columns ('cleaned_reviews' and 'sentiment_label').
    """
    input_ids = []
    attention_masks = []
    
    if is_inference and reviews is not None:
        for review in reviews:
            encoded_dict = tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='tf'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
    else:
        try:
            if data is not None:
                for review in data['cleaned_reviews']:
                    encoded_dict = tokenizer.encode_plus(
                        review,
                        add_special_tokens=True,
                        max_length=max_length,
                        truncation=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_tensors='tf'
                    )
                    input_ids.append(encoded_dict['input_ids'])
                    attention_masks.append(encoded_dict['attention_mask'])
        except:
            print("Input your dataset")


    labels = pd.get_dummies(data['sentiment_label']).to_numpy()  # One-hot encode labels

    return tf.convert_to_tensor(input_ids, dtype=tf.int32), tf.convert_to_tensor(attention_masks, dtype=tf.int64), tf.convert_to_tensor(labels)


def squeeze_tensors(input_ids, attention_masks):
    input_ids = tf.squeeze(input_ids, axis=1)
    attention_masks = tf.squeeze(attention_masks, axis=1)
    return input_ids, attention_masks

