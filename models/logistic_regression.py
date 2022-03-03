'''
 Original Researchers: Matthew Zhu, Yinan Zhao
 Research summarization and final model .py file creation done by: Maliha Lodi
 You can find the research done to create this model at this link: https://github.com/afnanrahman/EAFP/blob/main/notebooks/logistic_regression.ipynb
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.preprocessing import RobustScaler


def LogisticRegressionClassifier(data, hyperparameter_grid):

    no_attr = data.drop('attrition', axis=1)

    # We also scale the data to optimize the running time of our model.
    robust = RobustScaler()
    X = robust.fit_transform(no_attr)
    Y = data['attrition']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TRAIN_TEST_SPLIT_PCT, random_state=RANDOM_STATE)

    log_reg = LogisticRegression(max_iter=1000, random_state = RANDOM_STATE)
    
    #Hyper parameter tuning using grid search   
    grid = GridSearchCV(log_reg, hyperparameter_grid, verbose = 0, n_jobs=-1, error_score = 0.0)
    grid.fit(X_train, Y_train) 
    grid_predictions = grid.predict(X_test) 

    #Model Metrics
    acc = accuracy_score(Y_test, grid_predictions)
    cm = confusion_matrix(Y_test, grid_predictions)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    auc = roc_auc_score(Y_test, grid_predictions)
    fpr, tpr, thresholds = roc_curve(Y_test, grid_predictions)

    return log_reg, acc, precision, specificity, recall, f1, auc


if __name__ == "main":
    RANDOM_STATE = 42
    TRAIN_TEST_SPLIT_PCT = 0.3

    data = pd.read_csv("https://raw.githubusercontent.com/afnanrahman/EAFP/main/data/clean_smote_data.csv")

    #hyperparameters used in tuning
    param_grid = {'C': [0, 0.001, 0.005, 0.01, 0.03, 0.1, 1],  
              'solver': ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga'], 
              'penalty':['l1', 'l2', 'none', 'elasticnet']}

    logistic_regression = LogisticRegressionClassifier(data, param_grid)
    
    print('Accuracy: ', logistic_regression[1])
    print('Precision: ', logistic_regression[2])
    print('Specificity: ', logistic_regression[3])
    print('Recall: ', logistic_regression[4])
    print('F1 score: ', logistic_regression[5])
    print('AUC Score: ', logistic_regression[6])

