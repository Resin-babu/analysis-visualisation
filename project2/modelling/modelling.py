import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('delhivery_dataset50001.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['status'] = df['status'].str.strip().str.lower()
print('Unique status values:', df['status'].unique())


y = df['status'].map({'on-time': 0, 'delayed': 1})
print('Missing values in y:', y.isnull().sum())
df = df[~y.isnull()]
y = y[~y.isnull()]


features = ['route_type', 'actual_distance_to_destination', 'osrm_time', 'segment_actual_time',
            'segment_osrm_time', 'cutoff_factor', 'day&time', 'hour_of_day', 'time_of_day']
X= df[features]

X = pd.get_dummies(X, columns=['route_type', 'day&time', 'hour_of_day', 'time_of_day'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('Data processed and split successfully!')


X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


final_df = X.copy()
final_df['status'] = y


final_df.to_csv('combined_dataset.csv', index=False)

print('Combined CSV file saved successfully!')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)


print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)



plt.figure(figsize=(10,15))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='coolwarm')
plt.title('Feature Importance in Delay Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

