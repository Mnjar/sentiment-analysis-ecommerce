import json
import tensorflow as tf
from transformers import XLMRobertaTokenizer
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load hyperparameters
with open('config.json') as f:
    config = json.load(f)
    
def rnns_tokenizer(data, max_words=5000, max_len=250):
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(data['cleaned_reviews'].values)
    sequences = tokenizer.texts_to_sequences(data['cleaned_reviews'].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, tokenizer

def load_roberta_tokenizer(tokenizer_name):
    return XLMRobertaTokenizer.from_pretrained(tokenizer_name)

def tokenize_data(data, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    
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

    return tf.convert_to_tensor(input_ids, dtype=tf.int32), tf.convert_to_tensor(attention_masks, dtype=tf.int64), tf.convert_to_tensor(data['sentiment_label'].values)

def squeeze_tensors(input_ids, attention_masks):
    input_ids = tf.squeeze(input_ids, axis=1)
    attention_masks = tf.squeeze(attention_masks, axis=1)
    return input_ids, attention_masks

def tokenize_new_reviews(reviews, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

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

    return tf.convert_to_tensor(input_ids, dtype=tf.int32), tf.convert_to_tensor(attention_masks, dtype=tf.int64)

