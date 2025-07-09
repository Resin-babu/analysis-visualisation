from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')
df['ship_date'] = pd.to_datetime(df['ship_date'], format='%d/%m/%y')
df['shipping_delay'] = (df['ship_date'] - df['order_date']).dt.days

df_model = df[['segment', 'region', 'category', 'sub-category', 'shipping_delay', 'sales']]
df_encoded = pd.get_dummies(df_model, columns=['segment', 'region', 'category', 'sub-category'], drop_first=True)

X = df_encoded.drop('sales', axis=1)
y = df_encoded['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid_dt = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_dt, cv=5, scoring='r2')
grid_dt.fit(X_train, y_train)

print("Best Params (Decision Tree):", grid_dt.best_params_)
print("Best R² Score:", grid_dt.best_score_)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train)

print("Best Params (Random Forest):", grid_rf.best_params_)
print("Best R² Score:", grid_rf.best_score_)

