import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('global1.csv')

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month


yearly_sales = df.groupby('year')['sales'].sum().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x='year', y='sales', data=yearly_sales, palette='viridis')
plt.title('Yearly Sales')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()
monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
monthly_sales['order_date'] = monthly_sales['order_date'].astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(x='order_date', y='sales', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()
