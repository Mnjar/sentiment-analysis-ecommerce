import tensorflow as tf
from tensorflow.keras.layers import Embedding, SpatialDropout1D, GRU, LSTM, Dense, Bidirectional, Dropout
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from utils import log_evaluation_results, load_model
from tokenizer import rnns_tokenizer

def create_lstm_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = Embedding(input_dim=5000, output_dim=100)(inputs)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))(x)
    x = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_gru_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = Embedding(input_dim=5000, output_dim=100)(inputs)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))(x)
    x = Bidirectional(GRU(100, dropout=0.3, recurrent_dropout=0.3))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_rnns_model(model, learning_rate):
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), 
        metrics=['accuracy']
    )
    
def evaluate_rnns_model(model, test_input_ids, test_labels):
    # Predict labels for test data
    test_predictions = model.predict(test_input_ids)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    
    # Classification report
    report = classification_report(test_labels, test_pred_labels, target_names=['positive', 'negative', 'neutral'])
    print(report)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(test_labels, test_pred_labels)

    # Log evaluation results
    log_evaluation_results(report, conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['positive', 'negative', 'neutral'], 
                yticklabels=['positive', 'negative', 'neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# inference function
def rnns_inference(path, reviews: list):
    with open('../models/rnns_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    model = load_model(path)

    # Function to preprocess the input texts
    def preprocess_texts(texts, tokenizer, max_sequence_length):
        # Convert texts to sequences
        sequences = tokenizer.texts_to_sequences(texts)
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        return padded_sequences

    x, _ = rnns_tokenizer(pd.read_csv('../data/labeled_reviews_with_lexicon.csv'))
    max_sequence_length = x.shape[1]

    # Preprocess the new reviews
    input_data = preprocess_texts(reviews, tokenizer, max_sequence_length)

    # Make predictions
    predictions = model.predict(input_data)

    # Decode the predictions
    label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_labels = [label_mapping[np.argmax(prediction)] for prediction in predictions]

    # Print predictions
    for review, label in zip(reviews, predicted_labels):
        print(f"Review: {review}\nPredicted Sentiment: {label}\n")