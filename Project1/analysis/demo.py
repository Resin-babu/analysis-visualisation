import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('churn.csv')

plt.figure(figsize=(7,5))
sns.histplot(x='monthlycharges',data=df,kde=True,bins=28)
plt.title('distribution of monthly charges')
plt.xlabel('Monthly charges')
plt.ylabel('value')
plt.show()

sns.boxplot(x='churn', y='monthlycharges',data=df )
plt.title('monthly charges vs churn')
plt.xlabel('churn')
plt.ylabel('monthly charges')
plt.show()