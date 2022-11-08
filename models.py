from numpy import average
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import (precision_score, balanced_accuracy_score)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def get_MLP(params=None):
    if params == None:
        return MLPClassifier()
    
    return MLPClassifier(**params)

def get_GBC(params=None):
    if params == None:
        return GradientBoostingClassifier()

    return GradientBoostingClassifier(**params)

def get_RF(params=None):
    """
        Returns a Random Forest model with given parameters

    Args:
        params (dict): contains the parameters to pass to the model
    """
    if params == None:
        return RandomForestClassifier()

    return RandomForestClassifier(**params)

def get_DT(params=None):
    """
        Returns a Decision Tree model with specified parameters

    Args:
        params (dict): parameters
    """
    if params == None:
        return tree.DecisionTreeClassifier()

    return tree.DecisionTreeClassifier(**params)

def get_SVC(params=None):
    """
    Returns a SVC Model with specified parameters

    Args:
        params (dict): parameters
    """
    if params == None:
        return SVC()
    
    if "probability" in params.keys():
        return SVC(**params)

    return SVC(**params, probability=True)

def get_knn(params=None):
    """
    Returns a k-NN Model with specified parameters

    Args:
        params (dict): parameters
    """
    if params == None:
        return KNeighborsClassifier()

    return KNeighborsClassifier(**params)

def run_classifier(classifier, input_columns, output_column, test_input_col=None,\
     test_out_col=None, split=True):

    clf = None
    accuracy = None
    confusion = None
    roc = None

    if split:            
        #Split the data into Train (67%) and Test (33%) Set
        X_train, X_test, y_train, y_test = train_test_split(input_columns,\
            output_column, test_size = 0.33, random_state = 5, stratify=output_column)
    else:
        X_train = input_columns
        y_train = output_column
        X_test = test_input_col
        y_test = test_out_col
    
    #Train the Classifier
    
    classifier = classifier.fit(X_train, y_train)

    #Calculate Train and Test Accuracy
    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)

    #Generate Confusion Matrices
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    c_train = confusion_matrix(y_train, y_pred_train)
    c_test = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = c_test.ravel()
    confusion = [c_train, c_test]

    #Generate metrics required for ROC
    scores = classifier.predict_proba(X_test)
    fpr, tpr, thresh = metrics.roc_curve(y_test, scores[:, 1])
    roc =[
            fpr,
            tpr,
            thresh
    ]
    
    precision = precision_score(y_test, y_pred_test, average='binary')
    specificity = tn/(tn+fp)
    accuracy = [train_acc, test_acc, precision, specificity]
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    clf = classifier

    return {
        "Classifier": clf,
        "Confusion": confusion,
        "Accuracy": accuracy,
        "ROC": roc,
        "bal_acc": bal_acc
    }

