__author__ = "Marco Marson"
__version__ = "1.0"
__maintainer__ = "Marco Marson"
__email__ = "vollet.marson@gmail.com"
__status__ = "Development"

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from process import Process_Data
from compare import CompareAndSave

import xgboost as xgb

if __name__ == "__main__":
    pre_process = Process_Data()
    compare = CompareAndSave()

    pre_process.normalize()
    X_train, X_test, y_train, y_test = pre_process.get_train_test()

    svm =  SVC(gamma='auto', class_weight='balanced', random_state=10)
    svm.fit(X_train, y_train)

    neural_network =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                                            hidden_layer_sizes=(5, 2),
                                            random_state=10)
    neural_network.fit(X_train, y_train)

    # Testes SVM
    y_pred = svm.predict(X_test)
    compare.get_confusion_matrix_result(svm, "SVM", y_test, y_pred)

    # Testes Neural Network
    y_pred = neural_network.predict(X_test)
    compare.get_confusion_matrix_result(neural_network, "Neural_Network", y_test, y_pred)

    ##
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10, 100]}
    svc = SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)

    # Testes SVM with Grid Search
    y_pred = clf.predict(X_test)
    compare.get_confusion_matrix_result(clf, "SVM_WITH_GRID_SEARCH", y_test, y_pred)

    ## MODEL XGBOOST
    # xgb.XGBClassifier
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", max_depth=5, random_state=42)

    xgb_model.fit(X_train, y_train)

    # Testes SVM with Grid Search
    y_pred = xgb_model.predict(X_test)
    compare.get_confusion_matrix_result(xgb_model, "XGBOOST", y_test, y_pred)

    random_forest_algorithm = RandomForestClassifier(random_state=42)
    random_forest_algorithm.fit(X_train, y_train)

    # Testes RandomForestForest
    y_pred = random_forest_algorithm.predict(X_test)
    compare.get_confusion_matrix_result(random_forest_algorithm, "Random_Forest", y_test, y_pred)





