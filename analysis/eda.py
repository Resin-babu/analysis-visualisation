import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('churn.csv')
def plot_churn_distribution(df):
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x='churn', data=df, palette='Set2')
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='blue', xytext=(0, 10),
                    textcoords='offset points')
    
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.show()
    
plot_churn_distribution(df)

def plot_onlinesecurity(df):
    
    plt.figure(figsize=(12, 8))
    
    ab=sns.countplot(x='onlinesecurity', hue='churn', data=df, palette='Set2')
    total=len(df)
    for c in ab.patches:
        percentage=f'{100 * c.get_height() / total:.1f}%'
        ab.annotate(percentage,  (c.get_x() + c.get_width() / 2., c.get_height()), 
                    ha='center', va='center', fontsize=12, color='blue', xytext=(0, 10),
                    textcoords='offset points')
        
    
    plt.title('Churn by Online Security')
    plt.xlabel('Online Security')
    plt.ylabel('Count')
    plt.show()
plot_onlinesecurity(df)


def churn_pie_per_contract(df):
    ab=df['contract'].unique()
    df['churn']= df['churn'].map({'Yes': 'churned', 'No': 'active'})
    for contract_type in ab:
        churn_counts = df[df['contract'] == contract_type]['churn'].value_counts()

        plt.figure(figsize=(6, 6))
        plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
        plt.legend(labels=churn_counts.index, title="Churn Status", loc='best')
        plt.title(f'Churn Distribution for {contract_type} Contract')
        plt.show()

churn_pie_per_contract(df)


def contract_churn_bar(df):
    
    ab=sns.countplot(x='contract' , data=df, hue='churn', palette='Set2')
    total=len(df)
    for c in ab.patches:
        percentage=f'{100 * c.get_height()/total:.1f}%'
        ab.annotate(percentage, (c.get_x() + c.get_width() / 2., c.get_height()), 
                    ha='center', va='center', fontsize=12, color='blue', xytext=(0, 10),
                    textcoords='offset points')
    plt.title('Churn by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.show()
contract_churn_bar(df)
ax=sns.histplot(data=df, x='tenure', hue='churn', multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn Status (Percentage)')
plt.xlabel('Tenure')
plt.ylabel('counts')
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # Avoid labeling empty bars
        ax.annotate(f'{int(height)}',  # Show count as integer
                    (p.get_x() + p.get_width() / 2., height),  # Position: center of bar
                    ha='center', va='center', fontsize=9, color='black', xytext=(0, 5),
                    textcoords='offset points')
plt.show()


df['corr']=df['churn'].map({'Yes': '1','No': '0'})
corr=df.select_dtypes(include=['float64' , 'int64']).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

sns.countplot(x='seniorcitizen', hue='churn',data=df, palette='Set2')
plt.title('Churn by Senior Citizen Status')
plt.xlabel('Senior Citizen')
plt.ylabel('Churn Rate')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

df= pd.read_csv('churn.csv')
df['senior']=df['seniorcitizen'].apply(lambda  x: 'senior' if x== 1 else 'young')
df['churn'] = df['churn'].map({'Yes': 'Churned', 'No': 'Active'})
sv=df['senior']
for i in sv.unique():
    senior_churn=df[df['senior']==i]['churn'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(senior_churn, labels=senior_churn.index, autopct='%1.1f%%',colors=['red','skyblue'])
    plt.legend(labels=senior_churn.index, title="Churn Status", loc='best')
    plt.title(f'distribution - {i}')
    plt.show()
    
    
    
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