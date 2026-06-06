import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/Ames_Housing_Clean.csv")

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def hyperparameter_tunning_lasso(X_train_scaled, y_train):
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    grid_search = GridSearchCV(estimator = Lasso(max_iter = 10000), param_grid = param_grid, 
                               cv = 5, n_jobs = -1, scoring = 'r2')
    
    grid_search.fit(X_train_scaled, y_train)

    results = pd.DataFrame(grid_search.cv_results_)

    return results, grid_search.best_params_['alpha']

results, best_alpha = hyperparameter_tunning_lasso(X_train_scaled, y_train)
print("Best Alpha:", best_alpha)

def hyperparameter_tuning_ridge(X_train_scaled, y_train):
    param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    grid_search = GridSearchCV(
        estimator = Ridge(), param_grid = param_grid, cv = 5, 
        scoring = 'r2', return_train_score = True, n_jobs = -1)
    
    grid_search.fit(X_train_scaled, y_train)

    print(grid_search.best_params_) # Best alpha.
    print(grid_search.best_score_) # Best Average Cross Validation Score.

    results = pd.DataFrame(grid_search.cv_results_)
    
    return results, grid_search.best_params_['alpha']

results, best_alpha = hyperparameter_tuning_ridge(X_train_scaled, y_train)
print(f'The best alpha is {best_alpha}')

def train_ridge(X_test_scaled, X_train_scaled, y_train, best_alpha):
    model = Ridge(alpha = best_alpha)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    return model, predictions

ridge, ridge_pred = train_ridge(X_test_scaled, X_train_scaled, y_train, best_alpha)

def metrics_ridge(X_train, X_test, y_train, y_test, ridge_pred):
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

metrics_ridge(X_train_scaled, X_test_scaled, y_train, y_test, ridge_pred)

def train_lasso(X_test_scaled, X_train_scaled, y_train, best_alpha):
    model = Lasso(alpha = best_alpha)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    return model, predictions

lasso, lasso_pred = train_ridge(X_test_scaled, X_train_scaled, y_train, best_alpha)

def metrics_lasso(X_train, X_test, y_train, y_test, lasso_pred):
    rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    mae = mean_absolute_error(y_test, lasso_pred)
    r2score = r2_score(y_test,lasso_pred)
    print(f'The predictions are : {lasso_pred[:5]}')
    print(f'The mean value is : {y.mean()}')

    print(f'RMSE : {rmse}') # How much of the variation is explained by my model.
    print(f'MAE : {mae}') # On average how wrong is my model.
    print(f'R2 Score : {r2score}') # How much of the variation is explained by my model.

    # For training data : To check overfitting.
    lasso_pred_train = lasso.predict(X_train)
    r2score_train = r2_score(y_train, lasso_pred_train)
    print(f'R2 score for training : {r2score_train}')

metrics_lasso(X_train_scaled, X_test_scaled, y_train, y_test, lasso_pred)