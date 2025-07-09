import datetime as dt
import pandas as pd

df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)

reference_date = df['order_date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (reference_date - x.max()).days,  
    'order_id': 'nunique', 
    'sales': 'sum'         
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(rfm.head())


rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]) 
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(rfm.sort_values('RFM_Segment', ascending=False).head())

rfm.to_csv('rfm_customers_with_clusterss.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['ship_date'] = pd.to_datetime(df['ship_date'], dayfirst=True)

reference_date = df['order_date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (reference_date - x.max()).days,
    'order_id': 'nunique',
    'sales': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm.to_csv('rfm_customers_with_clusterss.csv', index=False)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments (Recency vs Frequency)')
plt.xlabel('Recency (Days)')
plt.ylabel('Frequency (Number of Orders)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segments (Frequency vs Monetary)')
plt.xlabel('Frequency (Number of Orders)')
plt.ylabel('Monetary (Total Sales)')
plt.legend()
plt.show()

cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'}).reset_index()
cluster_summary.to_csv('rfm_cluster_summary.csv', index=False)

print(cluster_summary)

plt.figure(figsize=(12, 7))
cluster_melted = cluster_summary.melt(id_vars='Cluster', var_name='Metric', value_name='Value')
sns.barplot(data=cluster_melted, x='Cluster', y='Value', hue='Metric')
plt.title('Cluster-wise Mean Values for RFM Metrics')
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.legend(loc='upper right')
plt.show()
