import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('delhivery_dataset5000.csv')

print(df.columns)

df['source_name'] = df['source_name'].fillna('Unknown')

print("Missing values after filling 'source_name':", df.isnull().sum())

cam=df[df['source_name'] == 'Unknown']
print(cam.head(20))


# Convert time columns to datetime using format='mixed'
time_columns = ['trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp']

for col in time_columns:
    df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
print("Data types after conversion:")
print(df.dtypes)

print("if any;",duplicates := df.duplicated().sum())

df['delivery_delay_time']=df['actual_time']-df['osrm_time']
df = df[df['delivery_delay_time'] >= 0]


df['status']=df['delivery_delay_time'].apply(lambda x:'delayed' if x > 5 else'on-time')

df['od_start_time']=pd.to_datetime(df['od_start_time'], errors='coerce')

df['day&time']= df['od_start_time'].dt.strftime('%a ,%y-%m-%d')

df['hour_of_day'] = df['od_start_time'].dt.hour

# Time of day categorization
def categorize_time(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Evening'

df['time_of_day'] = df['hour_of_day'].apply(categorize_time)

print(df[['data','time_of_day','day&time', 'hour_of_day']])

df.to_csv('delhivery_dataset50001.csv', index=False)

