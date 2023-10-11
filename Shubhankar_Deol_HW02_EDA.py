import pandas as pd

print("QUESTION 1.1\n")
df = pd.read_csv('breast-cancer-wisconsin.csv')
s = df.describe()
s = s.transpose()
print(s)

print("QUESTION 1.2\n")
missing = df.isna().sum()
print(missing)

print("QUESTION 1.3\n")
for i in range(1, 10):
    X = 'F' + str(i)
    df[X] = pd.to_numeric(df[X], errors='coerce')
df = df.fillna(df.mean())
print(df)

print("QUESTION 1.4\n")
print(pd.crosstab(df['Class'], df['F6'], margins=True))

print("QUESTION 1.5\n")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

scatter_matrix(df[['F1', 'F2', 'F3', 'F4', 'F5', 'F6']], alpha = 0.7, figsize=(10, 10), diagonal='hist')
plt.show()
plt.savefig('1.5_scatter_matrix.png')

print("QUESTION 1.6\n")
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.boxplot(df['F7'])
plt.title('F7')
plt.subplot(2,3,4)
plt.hist(df['F7'])
plt.title('F7')
plt.subplot(2,3,2)
plt.boxplot(df['F8'])
plt.title('F8')
plt.subplot(2,3,5)
plt.hist(df['F8'])
plt.title('F8')
plt.subplot(2,3,3)
plt.boxplot(df['F9'])
plt.title('F9')
plt.subplot(2,3,6)
plt.hist(df['F9'])
plt.title('F9')
plt.show()
plt.savefig('1.6_hist_box.png')

print("QUESTION 2\n")
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.csv')
for i in range(1, 10):
    X = 'F' + str(i)
    df[X] = pd.to_numeric(df[X], errors='coerce')
df = df.fillna(df.mean())
print(df)