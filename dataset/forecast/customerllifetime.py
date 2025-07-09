import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')

rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (df['order_date'].max() - x.max()).days,
    'order_id': 'nunique',
    'sales': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm['CLV'] = rfm['Frequency'] * rfm['Monetary']
plt.figure(figsize=(8, 5))
sns.histplot(rfm['CLV'], bins=30, kde=True, color='orange')
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV')
plt.ylabel('Customer Count')
plt.show()
