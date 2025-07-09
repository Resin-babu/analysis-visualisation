import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('delhivery_dataset50001.csv')

df['delivery_time'] = df['actual_time']-df['osrm_time']
df['status']=df['delivery_time'].apply(lambda x: 'delayed' if x > 5 else 'on-time').copy()
df.to_csv('delhivery_dataset50001.csv', index=False)

sns.countplot(data=df, x='status', palette='Set2')
plt.title('Delivery Status Distribution')
plt.xlabel('Delivery Status')
plt.ylabel('Count')
plt.savefig('fig1.png')
plt.show()

sns.countplot(x='route_type', hue='status',data=df, palette='Set1')
plt.title('Delivery Time by Route Type and Status')
plt.xlabel('Route Type')
plt.ylabel('Delivery Time')
plt.savefig('fig3.png')
plt.show()


sns.lineplot(x='od_start_time', y='delivery_time', data=df)
plt.title('Delivery Time Over Time')
plt.xlabel('Order Start Time')
plt.ylabel('Delivery Time')
plt.savefig('fig2.png')
plt.show()

sns.lineplot(x='od_start_time', y='delivery_time', hue='status', data=df)
plt.title('Delivery Time Over Time')
plt.xlabel('Order Start Time')
plt.ylabel('Delivery Time')
plt.savefig('fig4.png')
plt.show()

print(df.groupby(['route_type', 'status']).size())
on_time_rate = df['status'].value_counts(normalize=True)['on-time'] * 100
print(f'On-Time Delivery Rate: {on_time_rate:.2f}%')

print(df['delivery_delay_time'].isnull().sum())
