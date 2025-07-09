import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('global1.csv')
df['order_date'] = pd.to_datetime(df['order_date'], format='%d/%m/%y')


monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.lineplot(x='order_date', y='sales', data=monthly_sales, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


model = ExponentialSmoothing(monthly_sales['sales'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(6)

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['order_date'], monthly_sales['sales'], label='Actual Sales')
plt.plot(pd.date_range(monthly_sales['order_date'].max(), periods=6, freq='M'), forecast, label='Forecasted Sales', color='red', marker='o')
plt.title('Sales Forecast for Next 6 Months')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

forecast_df = pd.DataFrame({
    'Month': pd.date_range(monthly_sales['order_date'].max(), periods=6, freq='ME'),
    'Forecasted_Sales': forecast
})
forecast_df.to_csv('forecasted_sales.csv', index=False)


basket = df.groupby(['order_id', 'product_name'])['sales'].sum().unstack().reset_index().fillna(0).set_index('order_id')
basket = basket[basket.sum(axis=1) > 1]
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0).astype(bool)

frequent_itemsets = apriori(basket_sets, min_support=0.0005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules = rules.sort_values('lift', ascending=False)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())


rules_to_save = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
rules_to_save['antecedents'] = rules_to_save['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_to_save['consequents'] = rules_to_save['consequents'].apply(lambda x: ', '.join(list(x)))
rules_to_save.to_csv('top_association_rules.csv', index=False)
