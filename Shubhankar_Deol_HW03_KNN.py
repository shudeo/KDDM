import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin.csv')

for i in range(1, 10):
    X = 'F' + str(i)
    df[X] = pd.to_numeric(df[X], errors='coerce')
df = df.dropna()

X = df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

def run_knn(k):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)

for k in {3, 5, 10}:
    acc = run_knn(k)
    print('k =', k, 'score =', acc)