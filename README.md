# Product Review Classifier - Amazon Reviews Sentiment Analysis using NLP

This project implements an end-to-end **Natural Language Processing (NLP) pipeline** for sentiment analysis on the Amazon Appliances Reviews dataset.

The pipeline performs text preprocessing using **regex, lemmatization, and stopword removal**, extracts product features through **Part-of-Speech (POS) tagging** and **dependency parsing**, and converts text into numerical representations using **TF-IDF vectorization**. A **Logistic Regression** classifier is trained on the processed data, with **SMOTE** applied to address class imbalance. The model is evaluated using accuracy, F1-score, confusion matrix, and classification reports.

The project also includes **t-SNE visualization** of TF-IDF features and analysis of the most frequently discussed product features, providing insights into customer opinions and review sentiment.

## Tech Stack

- Python
- pandas
- NumPy
- NLTK
- spaCy
- scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib
