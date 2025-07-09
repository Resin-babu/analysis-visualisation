from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('churn.csv')

# Prepare Features and Target
X = df.drop(['customerid', 'churn'], axis=1)
y = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode Categorical Variables
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Dictionary
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC()
}

# Train and Evaluate Models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[model_name] = {'Accuracy': acc, 'F1 Score': f1}

# Display Results
print("\nModel Comparison Results:")
for name, metrics in results.items():
    print(f'{name} - Accuracy: {metrics["Accuracy"]:.4f}, F1 Score: {metrics["F1 Score"]:.4f}')

# 📊 Model Performance Visualization
plt.figure(figsize=(10, 6))
accuracy_scores = [metrics['Accuracy'] for metrics in results.values()]
f1_scores = [metrics['F1 Score'] for metrics in results.values()]
model_names = list(results.keys())

x = range(len(model_names))
plt.bar(x, accuracy_scores, width=0.4, label='Accuracy', color='skyblue', align='center')
plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1 Score', color='salmon', align='center')

plt.xticks([i + 0.2 for i in x], model_names, rotation=20)
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(axis='y')
plt.show()

# Convert results dictionary to DataFrame
model_performance_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
model_performance_df.columns = ['Model', 'Accuracy', 'F1 Score']

# Save to CSV for Tableau
model_performance_df.to_csv('model_performance_summary.csv', index=False)
print("\nModel performance summary saved as 'model_performance_summary.csv'")

