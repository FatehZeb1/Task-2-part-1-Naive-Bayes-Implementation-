# Task-2-part-1-Naive-Bayes-Implementation-
We'll use the TREC 2007 Spam Track public corpus, which consists of 50,000 emails labeled as spam or ham (non-spam).
We'll perform the following preprocessing steps:

Tokenization: split emails into individual words
Stopword removal: remove common words like "the", "and", etc.
Stemming: reduce words to their base form (e.g., "running" becomes "run")
Vectorization: convert emails into numerical vectors

CODE:

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
emails = pd.read_csv('spam.csv', encoding='latin-1')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails['text'], emails['label'], test_size=0.2, random_state=42)

# Create a CountVectorizer object
vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_count)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
