import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/Ames_Housing_Clean.csv")

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

ridge = Ridge(alpha = 10)
ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

def metrics(X_train, y_train, y_test, ridge_pred):
    rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    mae = mean_absolute_error(y_test, ridge_pred)
    r2score = r2_score(y_test, ridge_pred)
    print(f'The predictions are : {ridge_pred[:5]}')
    print(f'The mean value is : {y.mean()}')

    print(f'RMSE : {rmse}') # How much of the variation is explained by my model.
    print(f'MAE : {mae}') # On average how wrong is my model.
    print(f'R2 Score : {r2score}') # How much of the variation is explained by my model.

    # For training data : To check overfitting.
    ridge_pred_train = ridge.predict(X_train)
    r2score_train = r2_score(y_train, ridge_pred_train)
    print(f'R2 score for training : {r2score_train}')

metrics(X_train, y_train, y_test, ridge_pred)

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    grid_search = GridSearchCV(
        estimator = Ridge(), param_grid = param_grid, cv = 5, 
        scoring = 'r2', return_train_score = True, n_jobs = -1)
    
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_) # Best alpha.
    print(grid_search.best_score_) # Best Average Cross Validation Score.

    results = pd.DataFrame(grid_search.cv_results_)
    
    return results

result = hyperparameter_tuning(X_train, y_train)

def plot(results):
    plt.figure(figsize = (8, 5))
    plt.plot(results['param_alpha'], results['mean_test_score'], marker='o')

    plt.xlabel("Alpha")
    plt.ylabel("Cross Validation R²")
    plt.title("Ridge Alpha Tuning")

    plt.show()

plot(result)