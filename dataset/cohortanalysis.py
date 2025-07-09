import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from operator import attrgetter

df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')
df['order_month'] = df['order_date'].dt.to_period('M')
df['cohort_month'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
df['cohort_index'] = (df['order_month'] - df['cohort_month']).apply(attrgetter('n'))

cohort_data = df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)


retention = retention.iloc[-12:, :12]  


plt.figure(figsize=(14, 8))
sns.heatmap(retention, annot=True, fmt='.0%', cmap='YlGnBu', linewidths=0.5, linecolor='gray')
plt.title('Customer Retention by Monthly Cohort (Last 12 Months)')
plt.xlabel('Cohort Index (Months since First Purchase)')
plt.ylabel('Cohort Month')
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
