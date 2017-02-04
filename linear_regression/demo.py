import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read data
dataframe = pd.read_fwf('power_consumption.txt')
x_values = dataframe[['Horsepower']]
y_values = dataframe[['Consumption']]

regression = linear_model.LinearRegression()
regression.fit(x_values, y_values)

# Show fuel consumption per 100km with an engine that has 150 horsepower
print(regression.predict(150))

# Print graph
plt.scatter(x_values, y_values)
plt.plot(x_values, regression.predict(x_values))
plt.show()
