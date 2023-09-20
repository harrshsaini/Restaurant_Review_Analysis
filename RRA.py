import numpy as np
import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range(0,1000):
  review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  corpus.append(review)
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_trian, y_test = train_test_split(X,y,test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=501,criterion='entropy')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)
acc = round(model.score(x_test,y_test)*100,2)
print(str(acc)+'%')