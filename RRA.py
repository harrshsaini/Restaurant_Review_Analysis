import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score



ds=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

ds.shape,ds.columns

X_train, X_test, y_train, y_test = train_test_split(ds['Review'], ds['Liked'], test_size=0.2, random_state=42)



tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


svm_classifier = SVC(kernel='linear', C=1.0)

svm_classifier.fit(X_train_tfidf, y_train)

y_pred = svm_classifier.predict(X_test_tfidf)

y_pred

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)


precision =round(precision_score(y_test, y_pred)*100,2)
recall = round(recall_score(y_test, y_pred)*100,2)


print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


cp = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cp)

cp = confusion_matrix(y_test, y_pred)

precision = round(precision_score(y_test, y_pred)*100,2)
recall = round(recall_score(y_test, y_pred)*100,2)
accuracy = round(accuracy_score(y_test, y_pred)*100,2)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
