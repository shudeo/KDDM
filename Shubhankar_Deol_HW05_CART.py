import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.csv')

for i in range(1, 10):
    X = 'F' + str(i)
    df[X] = pd.to_numeric(df[X], errors='coerce')
df = df.dropna()

from sklearn.model_selection import train_test_split

X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
from sklearn.metrics import classification_report
rep = classification_report(y_test, y_pred, output_dict=False)
print('Report: ', rep)