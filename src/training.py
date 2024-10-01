import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tokenizer import load_roberta_tokenizer, tokenize_data, squeeze_tensors
from roberta_model import create_roberta_model, compile_roberta_model, evaluate_roberta_model
from rnns_model import create_lstm_model, create_gru_model, compile_rnns_model, evaluate_rnns_model
from utils import load_data, save_model

# Load hyperparameters
with open('config.json') as f:
    config = json.load(f)

# Load data using the helper function
train_data, val_data, test_data = load_data('./labeled_reviews_with_lexicon.csv', is_roberta_model=True)
x_train, x_val, y_train, y_val = load_data('./labeled_reviews_with_lexicon.csv', is_roberta_model=False)

def train_model(model, train_data, val_data, batch_size, epochs, patience):
    history = model.fit(
        x=train_data[0],  # Input features
        y=train_data[1],   # Labels
        validation_data=(val_data[0], val_data[1]),  # Validation data
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience)]
    )
    return history

def plot_training_history(history):
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
    
    # Tokenize for Roberta model
    train_input_ids, train_attention_masks, train_labels = tokenize_data(train_data, tokenizer, config['max_length'])
    val_input_ids, val_attention_masks, val_labels = tokenize_data(val_data, tokenizer, config['max_length'])
    test_input_ids, test_attention_masks, test_labels = tokenize_data(test_data, tokenizer, config['max_length'])

    # Squeeze tensors
    train_input_ids, train_attention_masks = squeeze_tensors(train_input_ids, train_attention_masks)
    val_input_ids, val_attention_masks = squeeze_tensors(val_input_ids, val_attention_masks)
    test_input_ids, test_attention_masks = squeeze_tensors(test_input_ids, test_attention_masks)

    # Create and compile model
    roberta_model = create_roberta_model(config['tokenizer_name'], config['num_labels'])
    lstm_model = create_lstm_model(input_shape=(config['tokenizer_name']), num_classes=(config['num_labels']))
    gru_model = create_gru_model(input_shape=(config['tokenizer_name']), num_classes=(config['num_labels']))
    
    compile_roberta_model(roberta_model, config['learning_rate'])
    compile_rnns_model(lstm_model, config['learning_rate'])
    compile_rnns_model(gru_model, config['learning_rate'])

    # Train model
    roberta_model_history = train_model(
        roberta_model, 
        (train_input_ids, train_attention_masks, train_labels),
        (val_input_ids, val_attention_masks, val_labels),
        config['batch_size'], 
        config['epochs'], 
        config['early_stopping_patience']
    )
    
    lstm_model_history = train_model(
        lstm_model, 
        (x_train, y_train),
        (x_val, y_val), 
        config['batch_size'], 
        config['epochs'], 
        config['early_stopping_patience']
    )

    gru_model_history = train_model(
        gru_model, 
        (x_train, y_train), 
        (x_val, y_val),
        config['batch_size'], 
        config['epochs'], 
        config['early_stopping_patience']
    )
    
    # Model summary
    roberta_model.summary()
    lstm_model.summary()
    gru_model.summary()

    # Save the trained model
    save_model(roberta_model, 'roberta_model.tf')
    save_model(lstm_model, 'lstm_model.tf')
    save_model(gru_model, 'gru.tf')

    # Evaluate model on test data
    evaluate_roberta_model(roberta_model, test_input_ids, test_attention_masks, test_labels)
    evaluate_rnns_model(lstm_model, test_input_ids, test_labels)
    evaluate_rnns_model(gru_model, test_input_ids, test_labels)
