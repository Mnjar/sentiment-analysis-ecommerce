# Sentiment Analysis With Self Made E-Commerce Dataset

## Project Overview

This project analyzes customer sentiment from an e-commerce platform using a self-made dataset. Dataset dikumpulkan menggunakan teknik web scrapping. It compares the performance of three models: RoBERTa, LSTM, and GRU, highlighting key differences.

The dataset was collected independently from one of the E-Commerce platforms. Scrapping technique was used to pursue efficiency in the data collection process.

## Model Used

- RoBERTa: Robustly Optimized Bert Pretraining Approach (RoBERTa) is a BERT (Bidirect Encoder Representation from Transformers) based transformer model, optimized to achieve better performance on NLP tasks. The model was developed by Facebook AI, and is designed to address some of BERT’s limitations by modifying the way the model is pretrained. One of the main differences is that RoBERTa removes the NSP (Next Sentence Prediction) task present in BERT. This task is considered less important for model performance, and its removal has been shown to improve results.

- LSTM (Long Short Term Memory): LSTM is a more complex version of RNN that was created to address the traditional RNN’s problem of handling long-term dependencies. LSTM introduces a special mechanism called **gate** to control the flow of information, allowing the model to retain important information for a long period of time and discard irrelevant information.

- GRU (Gated Recurrent Unit): GRU is a simpler variant of LSTM, this variant was introduced for architecture architecture by maintaining the ability to handle long-term problems and vanishing gradients that LSTM has.

## Dataset

The data collection process is carried out using the scrapping method with the selenium library and beautiful soup. The data source itself is taken from the comment section on the Tokopedia platform. To protect the privacy of stores and users, the data collection process does not include the name of the store and the name of the user who provided the review. The data collected only includes the product name, category, rating, and user reviews.

## Data Collections Process

1. Initialization and Page Loading

    The scrape_reviews function is responsible for collecting reviews from a given product page URL. Selenium is used to open the URL in a Chrome browser, emulating a user navigating to the page.

2. Extracting Store, Product, and Category Information

    Once the page loads, specific elements like the store name, product name, and product category are extracted using XPath selectors. These elements provide context for each review and are stored for every review associated with the product.

3. Scrolling to the Reviews Setion

    To ensure the reviews are loaded properly, a scroll action is performed to the review section of the page using JavaScript, allowing the reviews to render dynamically if necessary.

4. Handling Pagination

    The function detects pagination buttons on the page to determine the total number of pages of reviews. It iterates over each page, collecting reviews and their respective ratings.

5. Review and Rating Extraction

    Reviews are extracted using BeautifulSoup, searching for elements containing review text and star ratings. The star rating is parsed from the aria-label attribute of the rating element, and both the review text and rating are stored in a structured format.

    ```python
    def extract_rating(rating_element):
    aria_label = rating_element.get('aria-label', '')
    if "bintang" in aria_label:
        try:
            return int(aria_label.split(' ')[1])
        except ValueError:
            return None
    return None
    ```

6. Storing the Data

    Each review entry, including the store name, product name, category, review text, rating, and the URL from which it was scraped, is appended to a list. The process is repeated for each page of reviews.

## Data Cleaning and Preprocessing

The data cleaning and preprocessing pipeline was implemented to prepare the review text for sentiment analysis. This process involves several steps to ensure that the text data is clean, consistent, and ready for analysis. Below is an explanation of the workflow based on the code:

- Removing new lines characters
- Lowercasing text
- Removing emoticons
- Word normalization and Slang replacement
- Remove Stopwords
- Remove punctuation
- Remove number

## Evaluation Metrics

The evaluation metrics used to measure model performance are Recall, Precision, F1-Score, and Accuracy.

1. Precision

    Precision measures the proportion of correctly predicted positive sentiments out of all predictions made as positive.

    $$\text{PRECISSION} = \frac{TP}{TP+FP}$$

    Where TP is True Positives and FP is False Positives.
    In sentiment analysis, high precision means the model avoids labeling neutral or negative texts as positive sentiments.

2. Recall

    Recall measures the proportion of correctly predicted positive sentiments out of all actual positive sentiments in the dataset.

    $$\text{RECALL} = \frac{TP}{TP+FN}$$

    Where FN is false negatives.
    A high recall indicates that the model captures most of the positive sentiments, even if some predictions are incorrect.

3. F1 Score

    F1 Score is the harmonic mean of Precision and Recall, providing a balance between the two.

    $$\text{F1-Score} = 2 \cdot \frac{Precission \cdot Recall}{Precission + Recall}$$

    A high F1-Score shows that the model is effective at detecting positive sentiments while minimizing both false positives and false negatives.

4. Accuracy

    Accuracy measures the proportion of all correct predictions (positive, negative, and neutral) out of the total predictions.

    $$\text{Accuracy} = \frac{TP + TN}{TP+TN+FP+FN}$$

    While accuracy provides a general measure of model performance, it may not be reliable for imbalanced sentiment datasets where one class (e.g., neutral) dominates.

## Result

### RoBERTa

1. Training History (Accuracy & Loss)

    ![Training History](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/roberta/training-result-roberta.png?raw=true)

   This graph illustrates the performance of the model during the training process:
   - **Train Accuracy & Validation Accuracy**: The graph shows an increase in accuracy as the number of epochs increases, and at the end of training, these two metrics are almost parallel, indicating that the model is not experiencing significant overfitting or underfitting.

   - **Train Loss & Validation Loss**: The training and validation losses also decrease consistently. At the end of the epoch, the training and validation loss values ​​are close together, which also indicates that the model is well trained and generalizes well to new data.

2. Classification Report

    ![Classification Report](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/roberta/classification_report_roberta.png?raw=true)

    This classification report shows the model evaluation metrics, including precision, recall, f1-score, and support for each class (negative, neutral, and positive):

    - **Negative Class**:
        - Precision: 0.90, indicating that out of all predictions stated as negative, 90% were correct.

        - Recall: 0.89, indicating that out of all true negative examples, 89% were correctly identified by the model.

        - F1-score: 0.89, which is the harmonic mean of precision and recall, reflects a balance between the two.

    - **Neutral Class**:
        - Precision: 0.88, recall: 0.75, and f1-score: 0.81. These results indicate that the neutral class is slightly more difficult for the model to detect than the other classes, as can be seen from the lower recall.
    - **Positive Class**:
        - Precision: 0.96, recall: 0.98, and f1-score: 0.97, showing the best performance in this class, as positive class data may be more dominant in the dataset or easier for the model to recognize.

        - Accuracy: The model has an overall accuracy of 0.94 (94%), which is the percentage of data correctly predicted out of the total data.

        - Macro Avg and Weighted Avg provide an average of the overall metrics, taking into account the proportion of each class.

3. Confusion Matrix

    ![Confussion matrix](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/roberta/confussion-matrix-roberta.png?raw=true)

    The confusion matrix shows the number of correct and incorrect predictions from the model for each class:
     - **Negative Class**: Out of 261 data, 231 were correctly predicted as negative, 9 were incorrectly predicted as neutral, and 21 were incorrectly predicted as positive.

     - **Neutral Class**: Out of 155 data, 117 were correctly predicted as neutral, but 11 were incorrectly predicted as negative and 27 as positive. This shows the challenge of the model in recognizing the neutral class well.

     - **Positive Class**: Out of 1057 data, 1035 were correctly predicted as positive, while only 15 were incorrectly predicted as negative and 7 as neutral. This shows that the model is very accurate in recognizing the positive class.

---

### LSTM

1. Training History (Accuracy & Loss)

   ![Training History](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/lstm/training-result-lstm.png?raw=true)

   This graph illustrates the performance of the model during the training process:
   - **Train Accuracy & Validation Accuracy**: The graph shows a steady increase in accuracy as the number of epochs increases. The training and validation accuracies are almost parallel at the end of the training process, which indicates minimal overfitting or underfitting.

   - **Train Loss & Validation Loss**: Both training and validation losses decrease consistently. The loss values are close to each other at the end of the epoch, suggesting that the model is well-trained and generalizes effectively to unseen data.

2. Classification Report

   ![Classification Report](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/lstm/classification-report-lstm.png?raw=true)

   This classification report shows the evaluation metrics of the model, including precision, recall, f1-score, and support for each class (negative, neutral, and positive):

   - **Negative Class**:
     - Precision: 0.93, meaning that out of all the negative predictions, 93% were correct.
     - Recall: 0.86, indicating that out of all true negative examples, 86% were correctly identified by the model.
     - F1-score: 0.89, representing a balance between precision and recall.

   - **Neutral Class**:
     - Precision: 0.79, recall: 0.83, and f1-score: 0.81. These values show that the neutral class is more challenging for the model to detect, as reflected by the lower precision and f1-score.

   - **Positive Class**:
     - Precision: 0.97, recall: 0.98, and f1-score: 0.97. This shows the best performance for this class, likely because the positive class dominates the dataset or is easier for the model to recognize.

   - **Accuracy**: The overall accuracy of the model is 0.94 (94%), indicating that 94% of all predictions were correct.

   - **Macro Avg and Weighted Avg**: These averages provide a balanced view of the overall metrics, considering the class distribution in the dataset.

3. Confusion Matrix

   ![Confusion Matrix](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/lstm/confussion-matrix-lstm.png?raw=true)

   The confusion matrix shows the model's correct and incorrect predictions for each class:
   - **Negative Class**: Out of 498 data points, 428 were correctly predicted as negative, 40 were incorrectly predicted as neutral, and 30 were incorrectly predicted as positive.

   - **Neutral Class**: Out of 320 data points, 264 were correctly predicted as neutral, but 14 were incorrectly predicted as negative, and 42 as positive. This highlights the model's challenge in accurately identifying neutral sentiment.

   - **Positive Class**: Out of 2,113 data points, 2,086 were correctly predicted as positive, while only 15 were incorrectly predicted as negative and 12 as neutral. This shows that the model is highly accurate in recognizing positive sentiment.

---

### GRU

1. Training History

    ![Training History](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/gru/training-result-gru.png?raw=true)

    This graph illustrates the performance of the model during the training process:
    - **Train Accuracy & Validation Accuracy**: Similar to before, the graph shows a steady increase in accuracy as the number of epochs increases. The training and validation accuracies are almost parallel at the end of the training process, which indicates minimal overfitting or underfitting.

    - **Train Loss & Validation Loss**: Both training and validation losses decrease consistently. The loss values are close to each other at the end of the epoch, suggesting that the model is well-trained and generalizes effectively to unseen data, similar with LSTM model.

2. Classification Report

    ![Classification Report](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/gru/classification-report-gru.png?raw=true)

    This classification report summarizes the GRU model's evaluation metrics:

    - **Negative Class**:

        - Precision: 0.93, indicating that 93% of predictions for the negative class are correct.
        - Recall: 0.84, meaning the model correctly identifies 84% of all true negative samples.
        - F1-score: 0.88, reflecting a balance between precision and recall.
    - **Neutral Class**:

        - Precision: 0.80, recall: 0.78, and f1-score: 0.79. These metrics are lower compared to other classes, indicating that the neutral class is more difficult for the model to predict.

    - **Positive Class**:
        - Precision: 0.96, recall: 0.98, and f1-score: 0.97, showing the best performance for the positive class. This suggests that the model is highly effective at detecting positive samples, possibly due to a higher presence of positive examples in the dataset.

    - **Overall Accuracy**: 94%, meaning the model correctly predicts the sentiment for 94% of the data.

    - **Macro Avg** and **Weighted Avg** provide average metrics across all classes. The weighted avg takes into account the class distribution, while macro avg gives equal weight to all classes.

3. Confusion Matrix - GRU

    ![Confussion matrix](https://github.com/Mnjar/custom-sentiment-analysis-ecommerce/blob/main/result/gru/confussion-matrix-gru.png?raw=true)

    The confusion matrix provides a detailed view of the GRU model's prediction performance:

    - Negative Class: Out of 499 data points, 417 are correctly predicted as negative, while 39 are incorrectly classified as neutral and 43 as positive.

    - Neutral Class: Of the 318 neutral data points, 248 are correctly predicted. However, 20 were incorrectly predicted as negative, and 50 as positive. This highlights the model's challenge in accurately detecting neutral examples.

    - Positive Class: Out of 2128 positive samples, 2093 were correctly predicted, while only 13 were misclassified as negative, and 22 as neutral, indicating the model's strong performance in predicting positive sentiment.

### Key Observations

1. **Overall Accuracy**: All models show similar accuracy (~94%). However, accuracy alone isn't enough to choose a model, as class imbalance and specific needs (e.g., handling minority classes) should also be considered.

2. **Class Performance**:
   - **Positive Class**: All models show excellent performance in predicting positive sentiment with precision and recall near 96–98%, possibly because the dataset contains more positive samples.
   - **Neutral Class**: The neutral class appears to be the most difficult to predict across all models, especially for GRU and RoBERTa, where the recall and f1-scores are lower (between 0.75–0.81).
   - **Negative Class**: All models handle the negative class fairly well, but LSTM achieves slightly better balance between precision and recall.

3. **Training Stability**:
   - **RoBERTa**: Shows smooth and stable training with minimal overfitting. The training and validation losses are consistently close.
   - **LSTM**: Also demonstrates stable training, with parallel accuracy and loss trends across epochs.
   - **GRU**: Similarly stable, showing performance close to LSTM in terms of loss and accuracy throughout training.

### Recommendations

1. **Best for Class Imbalance**:
   If **handling class imbalance** is critical, especially for the neutral class, the **LSTM model** is recommended. It has higher precision (0.79) and recall (0.83) for the neutral class than RoBERTa and GRU, which suggests it handles minority class detection better. Additionally, LSTM shows a strong balance in performance across all classes.

2. **Best for Speed and Simplicity**:
   If computational efficiency and speed are priorities, **GRU** might be a better choice due to its simpler architecture and faster training compared to LSTM. Although it has a slightly lower recall for the neutral class, its overall accuracy and performance on positive and negative classes are similar to LSTM, making it a good trade-off between performance and training speed.

3. **Best for Robust Generalization**:
   For tasks where generalization and pre-trained language model features are essential (e.g., transfer learning with large datasets or more complex language features), **RoBERTa** is highly recommended. Despite slightly lower recall on the neutral class, RoBERTa’s architecture is more capable of capturing complex relationships in textual data, which may lead to better performance when fine-tuned further or applied to larger, more diverse datasets.

4. **Best for Scalability and Further Fine-tuning**:
   If you're planning to scale the model for production use, **RoBERTa** offers a flexible foundation for fine-tuning, particularly if there are plans to incorporate more complex or domain-specific data. It may also benefit from larger datasets or extended training.

### Final Choice

- **LSTM**: Best balance across all classes, especially for imbalanced datasets, and consistent training performance.

- **GRU**: Slightly faster and simpler, with good overall accuracy, but struggles slightly with neutral class prediction.

- **RoBERTa**: Ideal for more complex, generalizable tasks, but may require further fine-tuning to improve neutral class detection.

For immediate deployment and balanced performance, **LSTM** models are recommended unless speed or scaling is a critical factor.
