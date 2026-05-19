# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset from the provided Excel file
data = pd.read_excel(r"/content/supermarket_10years.xlsx")

# Create a 'Date' column by combining 'Year' and 'Month' and set it as index
data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%b')
data.set_index('Date', inplace=True)

# Assuming the target variable to forecast is 'Sales_Amount'
data.rename(columns={'Sales_Amount': 'sales_value'}, inplace=True)

# Drop original 'Year', 'Month', and other non-relevant columns for time series forecasting
data = data[['sales_value']]

# Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['sales_value'])
plt.xlabel('Date')
plt.ylabel('Sales Value') # Adjusted label
plt.title('Supermarket Sales Time Series') # Adjusted title
plt.show()

# Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna()) # .dropna() to handle potential NaNs
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['sales_value'])

# ACF and PACF plots
plot_acf(data['sales_value'].dropna())
plt.show()
plot_pacf(data['sales_value'].dropna())
plt.show()

# Split the data into train and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['sales_value'][:train_size], data['sales_value'][train_size:]

# SARIMA model configuration
# The seasonal_order (1, 1, 1, 12) assumes yearly seasonality for monthly data.
# Adjust this if your data frequency or seasonality is different.
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False) # disp=False to suppress convergence warnings/output

# Forecasting
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Sales Value') # Adjusted label
plt.title('SARIMA Model Forecasting for Supermarket Sales') # Adjusted title
plt.legend()
plt.show()
```

### OUTPUT:
<img width="1280" height="664" alt="image" src="https://github.com/user-attachments/assets/71c97f13-af54-49fc-98db-513438b65649" />
<img width="802" height="719" alt="image" src="https://github.com/user-attachments/assets/2a8e8375-e3a7-4298-ad8e-7d2f756015f2" />
<img width="1050" height="454" alt="image" src="https://github.com/user-attachments/assets/664d67e5-8f0b-43f1-af75-4ab00af2b6a3" />





### RESULT:
Thus the program run successfully based on the SARIMA model.
