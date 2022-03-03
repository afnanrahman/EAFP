'''
 Original Researchers: Kevin Qu, Omer Raza Khan
 Research summarization and final model .py file creation done by: Maliha Lodi
 You can find the research done to create this model at this link: https://github.com/afnanrahman/EAFP/blob/main/notebooks/support_vector_classifier.ipynb
'''

# Standard Packages
import pandas as pd
import numpy as np

# SVM and Model Metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

# Feature Selection
from sklearn.feature_selection import SequentialFeatureSelector

# Standardization
from sklearn.preprocessing import StandardScaler

# Model Selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


#Scaled, excluding 4 features, with C = 10 and Gamma = 0.1 using the default non-linear kernel
def SupportVectorClassifier(data):

    # Manually removing similar columns (in terms of context)
    remove = ['total_working_years', 'years_at_company', 'percent_salary_hike',
                'years_with_curr_manager', 'hourly_rate', 'daily_rate', 
                'monthly_rate']
    no_attrition = data.drop('attrition', axis=1)
    no_correlation = [e for e in list(no_attrition.columns) if e not in remove]

    # Scaling data using StandardScalar module
    sc = StandardScaler()
    sc.fit(data[no_correlation])
    data_scale = sc.transform(data[no_correlation])

    # Splitting data into train test sets
    X = data_scale
    Y = data['attrition']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TRAIN_TEST_SPLIT_PCT, random_state=RANDOM_STATE)
    clf = SVC(kernel='rbf', random_state=RANDOM_STATE)
    clf.fit(X_train, Y_train)

    #Feature Selection
    sfs = SequentialFeatureSelector(clf, n_features_to_select=19, direction="forward", scoring='accuracy')
    sfs = sfs.fit(X_train, Y_train)

    #Hyper parameter tuning
    param_grid = {"C": [0.01, 0.1, 1, 10, 100],
              "gamma": [0.01, 0.1, 1, 10, 100]}
    grid_cv = GridSearchCV(clf, param_grid, verbose=0)
    grid_cv.fit(X_train, Y_train)

    #Best hyperparameters
    print(grid_cv.best_estimator_)

    #Model Metrics
    grid_pred = grid_cv.predict(X_test)
    cm = confusion_matrix(Y_test, grid_pred, labels=grid_cv.classes_)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_cv.classes_)

    cm_disp.plot()
    print('Main Classification Metrics', classification_report(Y_test, grid_pred))

    # Working with the scaled data sets
    X_scale = pd.DataFrame(data_scale)
    Y_scale = pd.DataFrame(data["attrition"])
    Y_scale = np.ravel(Y_scale)

    kfold = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)

    # Computing metrics upon each fold for 5 folds 
    # Storing metrics in respective lists
    clf_accuracies = []
    clf_precisions = []
    clf_recall = []
    clf_f1_score = []

    for train_index, test_index in kfold.split(X):
        X_train_metrics, X_test_metrics = X_scale.iloc[train_index], X_scale.iloc[test_index] 
        y_train_metrics, y_test_metrics = Y_scale[train_index], Y_scale[test_index]
        clf.fit(X_train_metrics, y_train_metrics)
        y_pred_clf = clf.predict(X_test_metrics)

        accuracy_clf = accuracy_score(y_test_metrics, y_pred_clf)
        clf_accuracies.append(round(accuracy_clf, 4))

        precision = precision_score(y_test_metrics, y_pred_clf)
        clf_precisions.append(round(precision, 4))

        recall = recall_score(y_test_metrics, y_pred_clf)
        clf_recall.append(round(recall, 4))

        f1 = f1_score(y_test_metrics, y_pred_clf)
        clf_f1_score.append(round(f1, 4))
    
    auc = roc_auc_score(y_test_metrics, y_pred_clf)
    fpr, tpr, thresholds = roc_curve(y_test_metrics, y_pred_clf)

    return clf, clf_accuracies, clf_precisions, clf_recall, clf_f1_score, auc, cm_disp



if __name__ == "main":
    data = pd.read_csv("https://raw.githubusercontent.com/afnanrahman/EAFP/main/data/clean_smote_data.csv")
    RANDOM_STATE = 42
    TRAIN_TEST_SPLIT_PCT = 0.3

    svc = SupportVectorClassifier(data)

    print("Accuracy: %.2f%%" % (np.mean(svc[1])*100))
    print("Precisions: %.2f%%" % (np.mean(svc[2])*100))
    print("Recall: %.2f%%" % (np.mean(svc)*100[3]))
    print("F1 Score: %.2f%%" % (np.mean(svc[4])*100))
    print('AUC Score: ', svc[5])

