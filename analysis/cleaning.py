from sqlalchemy import create_engine
import pandas as pd

start=create_engine('mysql+mysqlconnector://root:Resin%402001@localhost:3306/churn')
df=pd.read_sql('SELECT * FROM churns', start)
print(df.head())

print(df.columns)

print(df.shape)
print(df.isnull().sum())
print('data type:',df['TotalCharges'].dtypes)

df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')

df.to_csv('churn.csv')

df['churn'] = df['churn'].map({'Yes': 'Churned', 'No': 'Active'})
print(df['churn'].value_counts())
print(df[df['churn'] == 'Churned'].head())

print(df[df['onlinesecurity'] == 'Yes']['churn'].value_counts())
print(df[df['onlinesecurity']== 'No']['churn'].value_counts())

print(df[df['onlinesecurity']=='Yes'].head())
print((df['onlinesecurity'] == 'yes').value_counts())
print(df['contract'].unique())

df['contract']= df['contract'].map({'Month-to-month':'Monthly','One year':'Yearly','Two year':'Two Years'})
print(df['contract'].value_counts())
