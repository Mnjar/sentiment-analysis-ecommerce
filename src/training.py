import json
import os
import datetime
import tensorflow as tf
from tf_keras.callbacks import TensorBoard, EarlyStopping
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow

from tokenizer import load_roberta_tokenizer, tokenize_data, squeeze_tensors
from models.roberta_model import create_roberta_model, compile_roberta_model, evaluate_roberta_model
from models.rnns_model import create_lstm_model, create_gru_model, compile_rnns_model, evaluate_rnns_model
from utils import load_data, save_model

# Load hyperparameters
with open('config.json') as f:
    config = json.load(f)

train_data, val_data, test_data = load_data('data/labeled_reviews_with_lexicon.csv', is_roberta_model=True)
x_train, y_train, x_val, y_val, X = load_data('data/labeled_reviews_with_lexicon.csv', is_roberta_model=False)


def train_model(model, train_data, val_data, batch_size, epochs, patience, model_name, log_dir, is_roberta_model: bool=False,):
    """
    Melatih model dan melogging eksperimen ke MLflow.

    Args:
        model: Model yang akan dilatih.
        train_data (tuple): Tuple berisi input dan label untuk training.
        val_data (tuple): Tuple berisi input dan label untuk validasi.
        batch_size (int): Ukuran batch.
                epochs (int): Jumlah epochs untuk training.
        patience (int): Jumlah epochs untuk early stopping jika tidak ada perbaikan.
        model_name (str): Nama dari model yang dilatih (misal 'Roberta', 'LSTM', 'GRU').

    Returns:
        history: Objek history dari proses training model.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    callbacks = []
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)
    
    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters ke MLflow
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('early_stopping_patience', patience)

        # Logging konfigurasi lainnya
        mlflow.log_param('model_name', model_name)
        # Mulai training model
        if is_roberta_model:
            # Mulai training model
            history = model.fit(
                x=[train_data[0], train_data[1]],  # Input features: [input_ids, attention_masks]
                y=train_data[2],  # Labels: one-hot encoded
                validation_data=([val_data[0], val_data[1]], val_data[2]),  # Validation data
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[EarlyStopping(monitor='val_loss', patience=patience), tensorboard_callback]
        )
        else:
            history = model.fit(
                x=train_data[0],  # Input features
                y=train_data[1],  # Labels
                validation_data=(val_data[0], val_data[1]),  # Validation data
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[EarlyStopping(monitor='val_loss', patience=patience), tensorboard_callback]
            )

        # Log metrics ke MLflow
        mlflow.log_metric('train_accuracy', history.history['accuracy'][-1])
        mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
        mlflow.log_metric('train_loss', history.history['loss'][-1])
        mlflow.log_metric('val_loss', history.history['val_loss'][-1])

        # Simpan model yang telah dilatih ke MLflow
        mlflow.tensorflow.log_model(model, model_name)

    return history

def plot_training_history(history):
    """
    Fungsi untuk menampilkan history training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()

if __name__ == "__main__":
    # Load tokenizer
    tokenizer = load_roberta_tokenizer(config['tokenizer_name'])
    
    # Tokenize data untuk model Roberta
    train_input_ids, train_attention_masks, train_labels = tokenize_data(train_data, tokenizer, config['max_length'])
    val_input_ids, val_attention_masks, val_labels = tokenize_data(val_data, tokenizer, config['max_length'])
    test_input_ids, test_attention_masks, test_labels = tokenize_data(test_data, tokenizer, config['max_length'])

    # Squeeze tensors untuk Roberta
    train_input_ids, train_attention_masks = squeeze_tensors(train_input_ids, train_attention_masks)
    val_input_ids, val_attention_masks = squeeze_tensors(val_input_ids, val_attention_masks)
    test_input_ids, test_attention_masks = squeeze_tensors(test_input_ids, test_attention_masks)

    # Buat dan compile model
    roberta_model = create_roberta_model(config['tokenizer_name'], config['num_labels'])
    lstm_model = create_lstm_model(input_shape=(X,), num_classes=(config['num_labels']))
    gru_model = create_gru_model(input_shape=(X,), num_classes=(config['num_labels']))
    
    # compile_roberta_model(roberta_model, config['learning_rate'])
    compile_rnns_model(lstm_model, config['learning_rate'])
    compile_rnns_model(gru_model, config['learning_rate'])

    # Train Roberta model
    roberta_log_dir = "logs/fit/roberta_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    lstm_log_dir = "logs/fit/lstm_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gru_log_dir = "logs/fit/gru_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    roberta_model_history = train_model(
        roberta_model, 
        (train_input_ids, train_attention_masks, train_labels),
        (val_input_ids, val_attention_masks, val_labels),
        config['batch_size'], 
        config['roberta_epochs'], 
        config['early_stopping_patience'],
        'Roberta',
        roberta_log_dir,
        True
    )
    
    # Train LSTM model
    lstm_model_history = train_model(
        lstm_model, 
        (x_train, y_train),
        (x_val, y_val), 
        config['batch_size'], 
        config['rnns_epochs'], 
        config['early_stopping_patience'],
        'LSTM',
        lstm_log_dir
    )

    # Train GRU model
    gru_model_history = train_model(
        gru_model, 
        (x_train, y_train), 
        (x_val, y_val),
        config['batch_size'], 
        config['rnns_epochs'], 
        config['early_stopping_patience'],
        'GRU',
        gru_log_dir
    )

    # Model summary
    roberta_model.summary()
    lstm_model.summary()
    gru_model.summary()

    # Save the trained models
    save_model(roberta_model, './models/roberta_model.tf')
    save_model(lstm_model, './models/lstm_model.tf')
    save_model(gru_model, './models/gru_model.tf')

    # Evaluate models on test data
    evaluate_roberta_model(roberta_model, test_input_ids, test_attention_masks, test_labels)
    evaluate_rnns_model(lstm_model, test_input_ids, test_labels)
    evaluate_rnns_model(gru_model, test_input_ids, test_labels)