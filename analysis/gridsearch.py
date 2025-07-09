from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('churn.csv')


X = df.drop(['customerid', 'churn'], axis=1)
y = df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],        
    'penalty': ['l1', 'l2'],       
    'solver': ['liblinear']         
}


grid_search_lr = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid_lr,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# Track time
start = time.time()
grid_search_lr.fit(X_train, y_train)
end = time.time()

print("\nBest Parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Score from GridSearch (Logistic Regression):", grid_search_lr.best_score_)
print(f"Time taken: {end - start:.2f} seconds")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time


param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# Track time
start = time.time()
grid_search_dt.fit(X_train, y_train)
end = time.time()

print("\nBest Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Score from GridSearch (Decision Tree):", grid_search_dt.best_score_)
print(f"Time taken: {end - start:.2f} seconds")


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=3)
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearch:", grid_search.best_params_)
print("Best Score from GridSearch:", grid_search.best_score_)


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('Classification Report:\n', classification_report(y_test, y_pred))


y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', color='green')
plt.title('Top 10 Features Impacting Churn')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_model, X, y, cv=5)
print('Cross-Validation Accuracy Scores:', scores)
print('Average Accuracy:', scores.mean())


X_test_with_results = X_test.copy()
X_test_with_results['Actual'] = y_test.values
X_test_with_results['Predicted'] = y_pred
X_test_with_results['Predicted_Probability'] = y_prob

X_test_with_results.to_csv('test_predictions.csv', index=False)

# Save Feature Importances
feature_importances_df = feature_importances.reset_index()
feature_importances_df.columns = ['Feature', 'Importance']

feature_importances_df.to_csv('feature_importance.csv', index=False)
# Save Cross-Validation Scores
cv_results_df = pd.DataFrame({'Fold': list(range(1, 6)), 'Accuracy': scores})

cv_results_df.to_csv('cross_validation_scores.csv', index=False)

# Manually create model summary
model_performance = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Best Accuracy': [
        grid_search_lr.best_score_,
        grid_search_dt.best_score_,
        grid_search.best_score_
    ]
}

model_performance_df = pd.DataFrame(model_performance)
model_performance_df.to_csv('model_performance_summary.csv', index=False)
