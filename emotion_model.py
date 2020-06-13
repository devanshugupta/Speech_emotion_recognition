# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
# Keras
#import keras
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC

import numpy as np
import pandas as pd
import scipy
import scipy.stats

data = pd.read_csv('H:/data3.csv')
data.head()
data = data.drop(['filename'], axis=1)

speech_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(speech_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = SVC(kernel='rbf',gamma='auto',C=5)
#model = MLPClassifier(hidden_layer_sizes=(400,),batch_size=256,learning_rate='adaptive',max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred[1],y_test[1])
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
print('-'*30)
print(confusion_matrix(y_test, y_pred))
print('*'*30)
print(classification_report(y_test, y_pred))