import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("data/Ames_Housing_Clean.csv")

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 42)

def hyperparameter_tunning(X_train, y_train):
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }

    grid_search = GridSearchCV(estimator = Lasso(max_iter = 10000), param_grid = param_grid, 
                               cv = 5, n_jobs = -1, scoring = 'r2')
    
    grid_search.fit(X_train, y_train)

    # print("Best Parameters:", grid_search.best_params_)
    # print("Best CV Score:", grid_search.best_score_)

    results = pd.DataFrame(grid_search.cv_results_)

    return results, grid_search.best_params_['alpha']

results, best_alpha = hyperparameter_tunning(X_train, y_train)
print("Best Alpha:", best_alpha)

lasso = Lasso(alpha = best_alpha, max_iter = 100000)
lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

def metrics(X_train, X_test, y_train, y_test, lasso_pred):
    rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    mae = mean_absolute_error(y_test, lasso_pred)
    r2score = r2_score(y_test, lasso_pred)
    print(f'The predictions are : {lasso_pred[:5]}')
    print(f'The mean value is : {y.mean()}')

    print(f'RMSE : {rmse}') # How much of the variation is explained by my model.
    print(f'MAE : {mae}') # On average how wrong is my model.
    print(f'R2 Score : {r2score}') # How much of the variation is explained by my model.

    # For training data : To check overfitting.
    lasso_pred_train = lasso.predict(X_train)
    r2score_train = r2_score(y_train, lasso_pred_train)
    print(f'R2 score for training : {r2score_train}')

metrics(X_train, X_test, y_train, y_test, lasso_pred)

def plot(results):
    plt.figure(figsize = (8, 5))
    plt.plot(results['param_alpha'], results['mean_test_score'], marker = 'o')

    plt.xlabel("Alpha")
    plt.ylabel("Cross Validation R²")
    plt.title("Lasso Alpha Tuning")

    plt.show()

plot(results)

def coef_survive(lasso):
    non_zero = np.sum(lasso.coef_ != 0)
    print(f'The features kept are : {non_zero}')
    print(f'The features removed are : {len(lasso.coef_) - non_zero}')

coef_survive(lasso)