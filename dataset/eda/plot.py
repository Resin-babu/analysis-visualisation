import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('global1.csv')

plt.figure(figsize=(12, 6))
sns.countplot(x='category',hue='segment',data=df, palette='Set2')
plt.title('Count of Categories by Segment')
plt.xticks(rotation=45)
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(title='Segment')
plt.show()

# Missing value count per column
print(df.isnull().sum())


plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()


df_cleaned = df.dropna()  



for col in ['sales']: 
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()


Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df['sales'] < (Q1 - 1.5 * IQR)) | (df['sales'] > (Q3 + 1.5 * IQR)))]


numeric_df = df.select_dtypes(include=['number'])


plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(6, 6))
plt.pie(df['sales'], labels=df['segment'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Sales Distribution by Customer Segment')
plt.show()


df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')

df['Year'] = df['order_date'].dt.year

yearly_sales = df.groupby('Year')['sales'].sum().reset_index()




top_products = df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10).reset_index()


plt.figure(figsize=(12, 8))
sns.barplot(data=top_products, y='product_name', x='sales', palette='coolwarm')
plt.title('Top 10 Products by Sales', fontsize=16)
plt.xlabel('Total Sales')
plt.ylabel('Product Name')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')
df['ship_date'] = pd.to_datetime(df['ship_date'], format='%d/%m/%y')


df['shipping_delay'] = (df['ship_date'] - df['order_date']).dt.days
plt.figure(figsize=(8, 5))
sns.histplot(df['shipping_delay'], bins=7, kde=True, color='coral')
plt.title('Shipping Delay Distribution')
plt.xlabel('Shipping Delay (Days)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(16, 8)) 
sns.barplot(x='category', y='sales', hue='sub-category', data=df, estimator=sum, ci=None)
plt.title('Sales by Category and Sub-Category', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)

plt.legend(title='Sub-Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  
plt.show()
