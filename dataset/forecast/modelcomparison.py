import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')
df['ship_date'] = pd.to_datetime(df['ship_date'], format='%d/%m/%y')
df['shipping_delay'] = (df['ship_date'] - df['order_date']).dt.days

df_model = df[['segment', 'region', 'category', 'sub-category', 'shipping_delay', 'sales']]
df_encoded = pd.get_dummies(df_model, columns=['segment', 'region', 'category', 'sub-category'], drop_first=True)

X = df_encoded.drop('sales', axis=1)
y = df_encoded['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))
    
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R² Score'])
print(results_df.sort_values(by='RMSE'))


import pandas as pd
import matplotlib.pyplot as plt

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_importance = pd.Series(dt_model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 5))
dt_importance.sort_values(ascending=False).head(10).plot(kind='barh', color='steelblue')
plt.title('Top 10 Feature Importances (Decision Tree)')
plt.gca().invert_yaxis()
plt.show()

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 5))
rf_importance.sort_values(ascending=False).head(10).plot(kind='barh', color='seagreen')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

results_df.to_csv('model_results.csv', index=False)
dt_importance.sort_values(ascending=False).to_csv('decision_tree_feature_importance.csv', header=['Importance'])
rf_importance.sort_values(ascending=False).to_csv('random_forest_feature_importance.csv', header=['Importance'])
