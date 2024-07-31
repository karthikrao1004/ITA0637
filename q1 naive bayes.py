import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)
data = {
    'battery_power': np.random.randint(500, 5000, 1000),
    'blue': np.random.randint(0, 2, 1000),
    'clock_speed': np.round(np.random.uniform(0.5, 3.0, 1000), 1),
    'dual_sim': np.random.randint(0, 2, 1000),
    'fc': np.random.randint(0, 20, 1000),
    'int_memory': np.random.randint(2, 64, 1000),
    'm_dep': np.round(np.random.uniform(0.1, 1.0, 1000), 1),
    'mobile_wt': np.random.randint(80, 200, 1000),
    'n_cores': np.random.randint(1, 8, 1000),
    'pc': np.random.randint(0, 20, 1000),
    'px_height': np.random.randint(0, 1960, 1000),
    'px_width': np.random.randint(500, 2000, 1000),
    'ram': np.random.randint(256, 8192, 1000),
    'sc_h': np.random.randint(5, 20, 1000),
    'sc_w': np.random.randint(0, 18, 1000),
    'talk_time': np.random.randint(2, 20, 1000),
    'price_range': np.random.randint(0, 4, 1000)
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())
print(df.dtypes)

if df.isnull().sum().sum()>0:
    for c in df.columns:
        df[c].fillna(df.mode()[0],inplace=True)
print(df)

X = df.drop('price_range', axis=1)
y = df['price_range']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(precision_score(y_test,y_pred,average='macro'))
print(recall_score(y_test,y_pred,average='macro'))
print(f1_score(y_test,y_pred,average='macro'))

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.show()

