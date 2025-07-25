# 📰 Fake News Detection 

This repository contains a deep learning model for detecting **fake news** using several word embedding techniques. The goal was to compare different methods for representing text, then train an effective classifier using the best-performing representation.

---

## 📁 Dataset

The dataset used is from Kaggle:

**[Fake and real news dataset – by George McIntire](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)**

It includes two CSV files:

- `True.csv` – Real news articles  
- `Fake.csv` – Fake news articles  

We combined and labeled them accordingly:
- `1` = fake
- `0` = real

---

## 🧹 Preprocessing

We performed the following preprocessing steps:

- Lowercased all text  
- Removed punctuation and special characters  
- Removed stopwords  
- Tokenized text  
- Padded sequences to equal lengths  

---

## 🧠 Word Embedding Techniques

We explored the following embedding approaches:

### 1. **Bag-of-Words (BoW)**  
- Used `CountVectorizer`  
- Very simple, fast, but ignores semantic meaning  
- Performs poorly on unseen data

### 2. **TF-IDF**  
- Term Frequency–Inverse Document Frequency  
- Weighs important words higher  
- Still sparse and lacks contextual understanding

### 3. **GloVe (Global Vectors)**  
- Pretrained word embeddings  
- Captures semantic relationships (e.g., king - man + woman ≈ queen)  
- Performed better than BoW/TF-IDF but didn't fully capture context

### 4. ✅ **Word2Vec (Final Choice)**  
- Captures both semantic and syntactic meaning  
- We trained our own Word2Vec model on the dataset  
- Embeddings were dense, continuous, and context-aware  
- Worked very well with neural networks and showed excellent results

---

## ✅ Final Model

We used a **Convolutional Neural Network (CNN)** over Word2Vec embeddings:

- **Embedding Layer**: Converts words to 100-dimensional vectors  
- **Conv1D**: Captures n-gram patterns  
- **GlobalMaxPooling1D**: Reduces dimension  
- **Dropout**: Prevents overfitting  
- **Dense layers**: For binary classification

---
## 📈 Performance

The model was trained and evaluated on an 80/20 train-test split.  
We monitored both **accuracy** and **loss** on the test data to ensure generalization.

### 🧪 Final Results:
- **Test Accuracy:** 99.88%  
- **Test Loss:** 0.0067  

These results indicate that the model is:
- Highly accurate in distinguishing fake from real news
- Not overfitting, as shown by the low loss and high accuracy on unseen data

### 📊 Classification Report :
```plaintext
        precision    recall  f1-score   support

           0       1.00      1.00      1.00      4241
           1       1.00      1.00      1.00      4568

    accuracy                           1.00      8809
    macro avg       1.00      1.00      1.00      8809
    weighted avg       1.00      1.00      1.00      8809
```
---
## ✅ Conclusion

This project successfully demonstrates the effectiveness of deep learning techniques in detecting fake news.  
By carefully preprocessing the data, experimenting with various embedding strategies, and tuning a CNN-based model, we achieved exceptional results with **99.88% accuracy** on unseen data.

The combination of **custom Word2Vec embeddings** and **1D convolutional layers** proved to be highly effective for this binary classification task.

Feel free to explore the notebook and reuse the techniques for similar NLP problems.

---

📬 For any suggestions or contributions, feel free to open an issue or pull request.

🧠 Built with ❤️ by Reem Kassab
