# **custom-sentiment-analysis-ecommerce**

## Project Overview

This project analyzes customer sentiment from an e-commerce platform using a self-made dataset. Dataset dikumpulkan menggunakan teknik web scrapping. It compares the performance of three models: RoBERTa, LSTM, and GRU, highlighting key differences.

Saya mengumpulkan dataset sendiri untuk melatih kemampuan saya dalam pengumpulan dataset dari nol. Teknik scrapping digunakan untuk mengejar efisiensi dalam proses pengumpulan data.

## Dataset

Proses pengumpulan data dilakukan menggunakan metode scrapping dengan libary selenium dan beautiful soup. Sumber data sendiri diambil dari komen section pada platform Tokopedia. Untuk melindungi privasi toko dan pengguna, proses pengumpulan data tidak menyertakan nama toko dan nama pengguna yang memberikan ulasan. Data yang dikumpulkan hanya mencakup nama produk, kategori, rating, dan ulasan pengguna.

## Model Used

- RoBERTa: Robustly Optimized Bert Pretraining Approach (RoBERTa) adalah sebuah model transformer berbasi BERT (Bidirectional Encoder Representation from Transformers), yang dioptimalkan untuk mencapai performa lebih baik pada tugas-tugas NLP. Model ini dikembangkan oleh Facebook AI, dan dirancang untuk memperbaiki beberapa keterbatasan BERT melalui modifikasi cara model dilatih. Salah satu perbedaan utama adalah RoBERTa menghapus tugas NSP (Next Sentence Prediction) yang ada di BERT. Tugas ini dinilai tidak terlalu penting untuk performa model, dan penghapusannya terbukti meningkatkan hasil.
- LSTM (Long Short Term Memory): LSTM adalah versi yang lebih kompleks dari RNN yang diciptakan untuk mengatasi masalah RNN tradisional dalam menangani dependensi jangka panjang. LSTM memperkenalkan mekanisme khusus yang disebut **gate** untuk mengontrol aliran informasi, sehingga memungkinkan model unuk menyimpan informasi yang penting dalam jangka waktu yang panjang dan membuang informasi yang tidak relevan.
- GRU (Gated Recurrent Unit): GRU adalah varian lebih sederhana dari LSTM, varian ini diperkenalkan untuk menyederhanakan arsitektur dengan mempertahankan kemampuan menangani masalah jangkan panjang dan vanishing gradient yang dimiliki LSTM.

## Evaluation Metrics

Metrik evaluasi yang digunakan untuk mengukur performa model adalah accuracy, F1-Score, precission, dan recall.

## Result

