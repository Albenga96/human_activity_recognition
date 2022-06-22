from utils.utils import basic_details
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore")


# Load datasets
train = pd.read_csv("data/train-1.csv")
test = pd.read_csv("data/test.csv")

train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
both['subject'] = '#' + both['subject'].astype(str)

print(basic_details(both))

activity = both['Activity']
label_counts = activity.value_counts()

plt.figure(figsize=(12, 8))
plt.bar(label_counts.index, label_counts)

plt.show()

Data = both['Data']
Subject = both['subject']
train = both.copy()
train = train.drop(['Data', 'subject', 'Activity'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(train, activity, test_size=0.2, random_state=0)

num_folds = 10
seed = 0
scoring = 'accuracy'
results = {}
accuracy = {}

model = KNeighborsClassifier(algorithm='auto', n_neighbors=8, p=1, weights='distance')

_ = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

accuracy["GScv"] = accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
