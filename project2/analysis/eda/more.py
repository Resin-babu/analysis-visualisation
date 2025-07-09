import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv('delhivery_dataset50001.csv')
sns.histplot(df['delivery_delay_time'], bins=40, kde=True)
plt.title('Delivery Time Distribution')
plt.xlabel('Delivery Time (Hours)')
plt.ylabel('Distributed')
plt.show()

sns.barplot(x='actual_time' , y='osrm_time' , data=df)
plt.title("delivery time overall")
plt.xlabel('actual time')
plt.ylabel('osrm time')
plt.show()
df['time_diff'] = df['actual_time'] - df['osrm_time']

sns.histplot(df['time_diff'], bins=30, kde=True)
plt.title('Difference Between Actual Time and Expected Time (OSRM)')
plt.xlabel('Time Difference (Actual - OSRM)')
plt.ylabel('Frequency')
plt.show()


top_sources = df['source_name'].value_counts().head(10).index
filtered_df = df[df['source_name'].isin(top_sources)]

sns.barplot(
    x='source_name',
    y='delivery_time',
    hue='status',
    data=filtered_df
)

plt.title('Top Source Centers vs Delivery Performance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(df['route_type'].isnull().sum())