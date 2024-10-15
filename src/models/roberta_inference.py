import tensorflow as tf
import json
from tokenizer import load_roberta_tokenizer, tokenize_data
from utils import load_model


# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(reviews, model, tokenizer, label_map, max_length=128):
    """
    Melakukan prediksi sentimen dari ulasan baru.

    Args:
    - reviews (list): List ulasan baru.
    - model: Model yang sudah dilatih untuk prediksi sentimen.
    - tokenizer: Tokenizer dari model.
    - label_map (dict): Mapping label sentimen.
    - max_length (int): Panjang maksimum token.

    Returns:
    - List berisi prediksi label sentimen untuk setiap ulasan.
    """
    # Tokenisasi ulasan baru
    input_ids, attention_masks = tokenize_data(reviews, tokenizer, max_length)
    
    # Squeeze untuk memastikan dimensi yang benar
    input_ids = tf.squeeze(input_ids, axis=1)
    attention_masks = tf.squeeze(attention_masks, axis=1)

    # Prediksi label sentimen
    predictions = model.predict([input_ids, attention_masks])
    pred_labels = tf.argmax(predictions.logits, axis=1)

    # Dekode hasil prediksi menjadi label sentimen
    pred_labels_decoded = [label_map[label] for label in pred_labels.numpy()]

    return pred_labels_decoded


# Fungsi utama untuk inference
def run_inference(model, tokenizer, new_reviews, label_map, max_length=128):
    """
    Menjalankan inferensi untuk prediksi sentimen pada ulasan baru.

    Args:
    - model: Model yang sudah dilatih.
    - tokenizer: Tokenizer dari model.
    - new_reviews (list): List ulasan baru.
    - label_map (dict): Mapping label sentimen.
    - max_length (int): Panjang maksimum token.

    Prints:
    - Hasil prediksi sentimen dari setiap ulasan.
    """
    # Prediksi sentimen ulasan baru
    predicted_labels = predict_sentiment(new_reviews, model, tokenizer, label_map, max_length)

    # Tampilkan hasil prediksi
    for review, label in zip(new_reviews, predicted_labels):
        print(f"Review: {review}\nPredicted Sentiment: {label}\n")


# Definisikan label sentimen
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

# Contoh ulasan baru
new_reviews = [
    "Produk ini sangat bagus dan saya suka", 
    "Produk sangat buruk, pengiriman lambat", 
    "kualitas barang lumayan, pengiriman cepat"
]

model = load_model('../models/roberta_model.')

with open('config.json') as f:
    config = json.load(f)

tokenizer = load_roberta_tokenizer(config['tokenizer_name'])


# Panggil fungsi inference utama
run_inference(model, tokenizer, new_reviews, label_map)