
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import (SMOTE, RandomOverSampler, ADASYN, SVMSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, NeighbourhoodCleaningRule, NearMiss, InstanceHardnessThreshold)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import make_classification
# from sklearn.model_selection import KFold


# def specificity(y_true, y_pred):
#     fpr, tpr, threshold = roc_curve(y_true, y_pred)
#     sp = 1 - fpr
#     return sp

scorer = make_scorer(recall_score,pos_label=0)

def hr():
    print("\n", "#"*100, "\n")

def get_cv_performance_metrics(classifier, X, y, cv, score=scorer):
    """Returns the accuracy of a classifier after n-fold cross-validation

    Args:
        classifier (sklearn.model): Classifier being used for the task
        X (np.array): Input Variables
        y (np.array): Target Variable
        cv (sklearn.model_selection): Sampler
    """
    scores = cross_val_score(classifier, X, y, scoring=score, cv=cv, n_jobs=-1)
    return {"Mean Score": mean(scores), "Std Dev": std(scores)}

def get_sampler(name, params):
    samplers= {
        "RandomOverSampler": RandomOverSampler,
        "ADASYN": ADASYN,
        "SVMSMOTE": SVMSMOTE,
        "SMOTE": SMOTE,
        "RandomUnderSampler": RandomUnderSampler,
        "NeighbourhoodCleaningRule": NeighbourhoodCleaningRule,
        "NearMiss": NearMiss,
        "InstanceHardnessThreshold": InstanceHardnessThreshold
    }
    return samplers[name](**params)

def plot_confusion_matrices(path, data, name):
    clf = data["Classifier"]
    confusion = data["Confusion"]
    train_acc, test_acc, precision, sp = data["Accuracy"]
    for i in range(len(confusion)):
        fig = ConfusionMatrixDisplay(confusion_matrix=confusion[0], display_labels=clf.classes_)
        fig.plot()
        fig.ax_.set_title(f"Train Accuracy: {train_acc}", fontsize=10)
        train_path = f"{path}/Train"

        if not os.path.isdir(train_path):
            os.makedirs(train_path)

        plt.savefig(f"{train_path}/{name}.jpg")
        
        fig = ConfusionMatrixDisplay(confusion_matrix=confusion[1], display_labels=clf.classes_)
        fig.plot()
        fig.ax_.set_title(f"Test Accuracy: {test_acc}\nPrecision: {precision}\nSpecificity (TNR): {sp}",\
            fontsize=8)
        test_path = f"{path}/Test"
        if not os.path.isdir(test_path):
            os.makedirs(test_path)
        plt.savefig(f"{test_path}/{name}.jpg")

def plot_roc(path, roc, name):
    path = f"{path}/roc_curves/"
    if not os.path.isdir(path):
        os.makedirs(path)
    #Extract data from roc variable
    fpr, tpr, thresh = roc
    
    #Generate and save plot
    plt.figure()
    plt.plot(fpr, tpr, lw = 2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"{path}/{name}")

