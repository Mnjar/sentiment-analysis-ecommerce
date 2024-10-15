import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import rnns_tokenizer
import pickle
import tensorflow as tf
import numpy as np


def load_data(file_path: str, is_roberta_model: bool=False):
    """Digunakan untuk load dataset

    Args:
        file_path (str): your dataset path
        is_roberta_model (bool, optional): IF true, pembagian dataset berjalan seperti biasa. Defaults to False.

    Returns:
        splitting: list yang mengandung dataset yang dibagi menggunakan train_test_split module.
    """
    data = pd.read_csv(file_path)
    data = data.dropna()
    sentiment_label = {'negative': 0, 'neutral': 1, 'positive': 2}
    data['sentiment_label'] = data['sentiment_label'].map(sentiment_label).astype(int)
    
    if is_roberta_model:
        # Split dataset: train, validation, test
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        return train_data, val_data, test_data
    else:
        # One-hot encode labels for RNN-based models (e.g., LSTM)
        Y = pd.get_dummies(data['sentiment_label']).to_numpy()
        X, tokenizer = rnns_tokenizer(data)
        # Split into training and temp (validation + test) data
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        #save tokenizer
        with open('models/tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return x_train, x_val, y_train, y_val, X.shape[1]

def log_evaluation_results(report, conf_matrix, path='evaluation_results.txt'):
    with open(path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))


def save_model(model, path):
    model.save(path)

def load_model(path):
    return tf.keras.models.load_model(path)