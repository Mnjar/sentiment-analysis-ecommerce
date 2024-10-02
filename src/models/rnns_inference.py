from rnns_model import rnns_inference

reviews = [
        "Produk ini sangat bagus dan saya suka", 
        "Layanan sangat buruk, tidak memuaskan.", 
        "Kualitas biasa saja, kurang memuaskan."
    ]

lstm_inference = rnns_inference('/models/gru_model.h5', reviews)
gru_inference = rnns_inference('/models/gru_model.h5', reviews)