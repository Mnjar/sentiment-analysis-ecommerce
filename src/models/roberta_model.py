import tensorflow as tf
from transformers import TFXLMRobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import log_evaluation_results

def create_roberta_model(tokenizer_name, num_labels):
    model = TFXLMRobertaForSequenceClassification.from_pretrained(tokenizer_name, num_labels=num_labels)
    
    model.roberta.trainable = False
    for layer in model.roberta.encoder.layer[-3:]:
        layer.trainable = True 
    
    return model

def compile_roberta_model(model, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

def evaluate_roberta_model(model, test_input_ids, test_attention_masks, test_labels):
    # Predict labels for test data
    test_predictions = model.predict([test_input_ids, test_attention_masks])
    test_pred_labels = tf.argmax(test_predictions.logits, axis=1)
    
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