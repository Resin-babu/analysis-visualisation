import pandas as pd
from sqlalchemy import create_engine


engine=create_engine('mysql+mysqlconnector://root:Resin%402001@localhost:3306/global')
df=pd.read_sql('select * from global1',engine)
print(df.head())

print('total:',df.shape)
print(df.info())

print(df.columns)

df.columns=df.columns.str.strip().str.lower().str.replace(' ', '_')
df.to_csv('global1.csv', index=False)

print(df['country'].unique())

print(df['country'].value_counts())
print(df[['country','order_date','sales','category','product_name','product_id']].isnull().sum())

print(df.describe())