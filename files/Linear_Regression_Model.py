import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 

df = pd.read_csv("data/Ames_Housing_Clean.csv")
# print(df.head())

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

linear_reg = LinearRegression();
linear_reg.fit(X_train, y_train)
y_pred  = linear_reg.predict(X_test)

def metrics(linear_reg, y_test, y_pred, X_test, X_train):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2score = r2_score(y_test, y_pred)
    print(f'The predictions are : {y_pred[:5]}')
    print(f'The mean value is : {y.mean()}')

    print(f'RMSE : {rmse}') # How much of the variation is explained by my model.
    print(f'MAE : {mae}') # On average how wrong is my model.
    print(f'R2 Score : {r2score}') # How much of the variation is explained by my model.

    # For training data : To check overfitting.
    y_pred_train = linear_reg.predict(X_train)
    r2score_train = r2_score(y_train, y_pred_train)
    print(f'R2 score for training : {r2score_train}')

metrics(linear_reg, y_test, y_pred, X_test, X_train)

def plots(y_test, y_pred):
    plt.figure(figsize = (8, 6)) # Size of plot.
    plt.scatter(y_test, y_pred, alpha = 0.5) # Alpha controls transparency. 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # (x-coordinates, y-coordinates, red line dashed)

    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted Prices")

    residuals = y_test - y_pred
    plt.figure(figsize = (8, 6))
    plt.scatter(y_pred, residuals, alpha = 0.5)
    plt.axhline(0, color='red', linestyle='--') # Draw a horizontal line at y = 0.

    # Negative -> Model predicted too high.
    # Positive residual means -> Model predicted too low.

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    plt.show()

plots(y_test, y_pred)

# To check for outliers.
def outliers(error):
    errors['Residuals'] = errors['Actual'] - errors['Predicted']
    errors['AbsResidual'] = errors['Residuals'].abs()
    df2 = errors.sort_values('AbsResidual', ascending=False).head(10)
    print(df2)

errors = pd.DataFrame({
        'Actual' : y_test,
        'Predicted' : y_pred
    })

outliers(errors)
