
from numpy import mean
from numpy import std
import numpy as np
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import (SMOTE, RandomOverSampler, ADASYN, SVMSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, NeighbourhoodCleaningRule, NearMiss, InstanceHardnessThreshold)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (roc_curve, make_scorer, recall_score, \
    balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, fbeta_score)



# def specificity(y_true, y_pred):
#     fpr, tpr, threshold = roc_curve(y_true, y_pred)
#     sp = 1 - fpr
#     return sp

scorer = make_scorer(recall_score,pos_label=0)

def f2_score(y_true, y_pred):
    f2_score = fbeta_score(y_true, y_pred, beta=2)
    return f2_score

def f2_scorer():
    return make_scorer(f2_score)

def k_fold_cv(classifier, sampler,  X, y, cv, score_name="accuracy"):
    scorers = {
        "accuracy": accuracy_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f2": fbeta_score
    }

    scores = []
    for train_folds_idx, val_fold_idx in cv.split(X, y):
        X_train_fold, y_train_fold = X[train_folds_idx], y[train_folds_idx]
        X_val_fold, y_val_fold = X[val_fold_idx], y[val_fold_idx]
        if sampler is None:
            X_sampled_train, y_sampled_train = X_train_fold, y_train_fold
        else:
            X_sampled_train, y_sampled_train = sampler.fit_resample(X_train_fold, y_train_fold)
        model_obj = classifier.fit(X_sampled_train, y_sampled_train)
        if score_name == "f2":
            score = scorers[score_name](y_val_fold, model_obj.predict(X_val_fold), beta = 2)
        else:
            score = scorers[score_name](y_val_fold, model_obj.predict(X_val_fold))
        scores.append(score)

    return np.array(scores).mean()



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

def plot_confusion_matrices(path, data, name, cv_score=None, scorer_name = "Balanced Accuracy"):
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
        fig.ax_.set_title(f"Test Accuracy: {test_acc}\n {scorer_name} on CV: {cv_score}\n\
            Precision: {precision}\nBalanced Accuracy: {data['bal_acc']}", fontsize=8)
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

