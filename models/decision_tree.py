'''
 Original Researcher: Maliha Lodi
 Research summarization and final model .py file creation done by: Maliha Lodi
 You can find the research done to create this model at this link: https://github.com/afnanrahman/EAFP/blob/main/notebooks/decision_tree.ipynb
'''

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score


def DecisionTreeClassifier(data):

    X = data.drop('attrition', axis=1)
    Y = data['attrition']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TRAIN_TEST_SPLIT_PCT, random_state=RANDOM_STATE)

    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    sfs_f = SequentialFeatureSelector(decision_tree, n_features_to_select=9, direction="forward", scoring='accuracy')
    sfs_f = sfs_f.fit(X_train, Y_train)

    cols = sfs_f.get_support(indices=True)
    feat_sfs = X_train.iloc[:, cols].columns

    X_train_sfs = X_train[feat_sfs]
    decision_tree.fit(X_train_sfs, Y_train)

    X_test_sfs = X_test[feat_sfs]
    Y_pred_sfs = decision_tree.predict(X_test_sfs)

    acc = accuracy_score(Y_test, Y_pred_sfs)
    precision = precision_score(Y_test, Y_pred_sfs)
    recall = recall_score(Y_test, Y_pred_sfs)
    f1 = f1_score(Y_test, Y_pred_sfs)

    cm = confusion_matrix(Y_test, Y_pred_sfs)
    auc = roc_auc_score(Y_test, Y_pred_sfs)

    return decision_tree, acc, precision, recall, f1, cm, auc


if __name__ == "main":
    RANDOM_STATE = 42
    TRAIN_TEST_SPLIT_PCT = 0.3

    data = pd.read_csv("https://raw.githubusercontent.com/afnanrahman/EAFP/main/data/clean_smote_data.csv")

    decision_tree = DecisionTreeClassifier(data)
    
    print('Accuracy: ', decision_tree[1])
    print('Precision: ', decision_tree[2])
    print('Recall: ', decision_tree[3])
    print('F1 score: ', decision_tree[4])
    print('AUC Score: ', decision_tree[5])

    









if __name__ == "main":
    data = pd.read_csv("https://raw.githubusercontent.com/afnanrahman/EAFP/main/data/clean_smote_data.csv")
    RANDOM_STATE = 42
    TRAIN_TEST_SPLIT_PCT = 0.3

    # logistic_regression = LogisticRegressionClassifier(data)
    
    # print('Accuracy: ', logistic_regression[1])
    # print('Precision: ', logistic_regression[2])
    # print('Specificity: ', logistic_regression[3])
    # print('Recall: ', logistic_regression[4])
    # print('F1 score: ', logistic_regression[5])
    # print('AUC Score: ', logistic_regression[6])