import pandas as pd

df=pd.read_csv('data/delhivery_data.csv')

print(df.shape)

small_df=df.head(5000)

small_df.to_csv('delhivery_dataset5000.csv', index=False)
print(small_df.shape)


print("missing values: ",small_df.isnull().sum())