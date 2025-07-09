from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv('churn.csv')

X=df.drop(['customerid' , 'churn'], axis=1)
y=df['churn'].apply(lambda x: 1 if x=='Yes' else 0)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X.to_csv('churndup.csv', index=False)
print(y.values[:15])