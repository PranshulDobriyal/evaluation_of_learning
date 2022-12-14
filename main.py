import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from models import (get_RF, get_DT, get_knn, get_SVC, run_classifier)
from helper_functions import (hr,\
     get_sampler, plot_confusion_matrices, plot_roc, k_fold_cv)
from sklearn.model_selection import GridSearchCV
import pprint 
import sys
import pickle
from sklearn.metrics import (make_scorer, recall_score)
from sklearn.preprocessing import quantile_transform

scorer = make_scorer(recall_score,pos_label=0)
pp = pprint.PrettyPrinter(indent=4)


#Load the data
data = pd.read_csv("data/drug_consumption.data", header=None)
D_input_cols = data.iloc[:, 1:13]
# We are only concerned with 'Alcohol' Class
D_target = data.iloc[:, 13]
# D_input_cols and D_target together form the dataset D

#Convert C1 (CL0) and C2 (CL1) to non-user (0) and all other classes to user (1)
D_target.loc[(D_target == "CL1") | (D_target == "CL0")] = 0
D_target.loc[D_target != 0] = 1

#Feature Selection

transformer = GenericUnivariateSelect(mutual_info_classif, mode='k_best', param=11)
cols = D_input_cols.columns
D_input_cols = transformer.fit_transform(D_input_cols, D_target.astype(int))
#print("Retained Features: ", transformer.get_feature_names_out(cols))

#Scaling Data
D_input_cols = quantile_transform(D_input_cols)

#Convert to numpy.array
D_target = np.array(D_target).astype('int')
D_input_cols = np.array(D_input_cols)

#########################################################################################################
# TASK 0: Rerun all the algorithms and use 10-fold CV
#########################################################################################################

hr()
print("\t\t\tRunning 10-fold CV using all 4 classifiers.")

param_dict = [
    {"min_samples_leaf": 2},
    {"max_depth": 5, "random_state": 0},
    {},
    {"n_neighbors": 5}]

classifiers = [get_DT, get_RF, get_SVC, get_knn]
names = ["DecisionTree", "RandomForest", "SVC", "k-NN"]


#initialise the object for CV sampler
cv = KFold(n_splits=10, random_state=1, shuffle=True)
strat_cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
acc= []
strat_acc = []
for i in range(len(classifiers)):
    classifier = classifiers[i](param_dict[i])
    score = k_fold_cv(classifier, None, D_input_cols, D_target, cv, score_name="accuracy")
    strat_score = k_fold_cv(classifier, None, D_input_cols, D_target, strat_cv, score_name="accuracy")
    acc.append(score)
    strat_acc.append(strat_score)

for i in range(len(names)):
    hr()
    print(names[i])
    print(f"\nMean Balanced Accuracy for 10 fold CV: {acc[i]}")
    print(f"\nMean Balanced Accuracy for Stratified 10 fold CV: {strat_acc[i]}")
hr()


#########################################################################################################
                            # TASK 1&2: Oversampling and Training
#########################################################################################################


# We will use 4 samplers with multiple different configurations each to get metrics for the 4 algorithms
sampler_names = ["RandomOverSampler", "ADASYN", "SVMSMOTE", "SMOTE"]
sampler_metrics = {}

params = [
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1]},
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "n_neighbors": [2, 3, 5, 7, 9]},
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "k_neighbors": [2, 3, 5, 7, 9]},
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "k_neighbors": [2, 3, 5, 7, 9]}
]

X_train, X_test, y_train, y_test = train_test_split(D_input_cols,\
     D_target, test_size = 0.33, random_state = 5)

# Change this to introduce test set if needed.
X = X_train
y = y_train

# Change this if stratified KFold is preferred
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

def generate_metric_matrix():
    #Generate comparision matrix
    #Loop over all possible combinations of Models->Sampler->Parameters
    for i in range(len(names)):
        sampler_metrics[names[i]] = {}
        classifier = classifiers[i](param_dict[i])
        for name, param in zip(sampler_names, params):
            keys = list(param.keys())
            if name not in sampler_metrics[names[i]].keys():
                sampler_metrics[names[i]][name] = {"Score": [], "param": []}

            if len(keys) == 1:
                for val in param[keys[0]]:
                    sampler = get_sampler(name, {keys[0]: val})
                    sampler_metrics[names[i]][name]["Score"].append(k_fold_cv(classifier, sampler,\
                         X, y, strat_cv, "balanced_accuracy"))
                    sampler_metrics[names[i]][name]["param"].append({keys[0]: val})

            else:
                for v1 in param[keys[0]]:
                    for v2 in param[keys[1]]:
                        print("-", end=" ")
                        sys.stdout.flush()
                        sampler = get_sampler(name, {keys[0]: v1, keys[1]: v2})
                        sampler_metrics[names[i]][name]["param"].append({keys[0]: v1, keys[1]: v2})
                        sampler_metrics[names[i]][name]["Score"].append(k_fold_cv(classifier, \
                            sampler, X, y, strat_cv, "balanced_accuracy"))

    # Save the dictionary to file
    with open('pickled_objects/metric_dict.pkl', 'wb') as f:
        pickle.dump(sampler_metrics, f)

#Run the below command only when metric_matrix is to be generated

#generate_metric_matrix()


############################################## Construct DB1  ################################################

# We know from the generated matrix that SMOTE with a sampling_strategy of 1 and k_neighbours = 7
#  gives the best balanced_accuracy on D
sampler = get_sampler("SMOTE", {"sampling_strategy": 1, "k_neighbors": 7})

DB1_X, DB1_y = sampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(DB1_X,\
     DB1_y, test_size = 0.33, random_state = 5)

# Retraining using DB1
hr()
print("\tTraining Models using DB1")
hr()

#Constructing parameter dictionary for hyperparameter tuning
classifier_param_dict = [
    {
        "min_samples_leaf": [2, 3, 4, 5],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [2, 3, 5],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.3, 0.5],
        "max_features": ["sqrt", "log2"]
        },
    {
        "max_depth": [3, 5, 7],
        "random_state": [0], 
        "criterion": ["gini", "entropy", "log_loss"],
        "min_weight_fraction_leaf": [0.0, 0.1, 0.3, 0.5],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False]
        },
    {
        "kernel": ("linear", "poly", "rbf"),
        "degree": [1, 2, 3],
        "random_state": [0],
        "C": [0.01, 0.1, 1, 10],
        "probability": [True],
        },
    {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "leaf_size": [10, 30, 50],
        "p": [1, 2, 3]
        }]

classifiers = [get_DT, get_RF, get_SVC, get_knn]
names = ["DecisionTree", "RandomForest", "SVC", "k-NN"]


#initialise the object for CV sampler
acc= []
std = []
def run_search():
    best_params = {} 
    for i in range(len(classifiers)):
        # Using GridSearch to get the best combination of parameters
        classifier = GridSearchCV(classifiers[i](), classifier_param_dict[i], verbose=1, scoring="balanced_accuracy")
        classifier.fit(X_train, y_train)
        best_params[names[i]] = classifier.best_params_

    with open('pickled_objects/best_over_param_dict.pkl', 'wb') as f:
        pickle.dump(best_params, f)

# Run the function below to generate a dictionary containing the best parameters for all the classifiers
run_search()

# Load the best parameters
with open('pickled_objects/best_over_param_dict.pkl', 'rb') as f:
    best_param = pickle.load(f)


############################################## Retraining the 4 classifiers with DB1 ###############################################
def retrain_on_db1():
    for i in range(len(classifiers)):
        name = names[i]
        classifier = classifiers[i](best_param[names[i]])
        met = run_classifier(classifier, X_train, y_train, X_test, y_test, split=False)
        cv_score = k_fold_cv(classifier, None, DB1_X, DB1_y, strat_cv, "balanced_accuracy")
        plot_confusion_matrices("Plots/algos_on_DB1", met, name, cv_score)
        plot_roc("Plots/algos_on_DB1", met["ROC"], name)

retrain_on_db1

#########################################################################################################
                            # TASK 3&4: Undersampling and Training
#########################################################################################################

sampler_names = ["RandomUnderSampler", "NeighbourhoodCleaningRule", "NearMiss", "InstanceHardnessThreshold"]
sampler_metrics = {}

param_dict = [
    {"min_samples_leaf": 2},
    {"max_depth": 5, "random_state": 0},
    {},
    {"n_neighbors": 3}]

params = [
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "replacement": [True, False]},
    {"n_neighbors": [2, 3, 5, 7], "kind_sel": ["mode", "all"]},
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "n_neighbors": [2, 3, 5, 7, 9]},
    {"sampling_strategy": [0.3, 0.5, 0.7, 0.9, 1], "cv": [3, 5, 7, 9]}
]

def generate_undersampling_metric_matrix():
    #Generate comparision matrix
    #Loop over all possible combinations of Models->Sampler->Parameters
    for i in range(len(names)):
        sampler_metrics[names[i]] = {}
        classifier = classifiers[i](param_dict[i])
        for name, param in zip(sampler_names, params):
            keys = list(param.keys())
            if name not in sampler_metrics[names[i]].keys():
                sampler_metrics[names[i]][name] = {"Score": [], "param": []}

            for v1 in param[keys[0]]:
                for v2 in param[keys[1]]:
                    print("-", end=" ")
                    sys.stdout.flush()
                    sampler = get_sampler(name, {keys[0]: v1, keys[1]: v2})
                    sampler_metrics[names[i]][name]["Score"].append(k_fold_cv(classifier, sampler,\
                         X, y, strat_cv, "balanced_accuracy"))
                    sampler_metrics[names[i]][name]["param"].append({keys[0]: v1, keys[1]: v2})
                    
    # Save the dictionary to file
    with open('pickled_objects/undersampling_metric_dict.pkl', 'wb') as f:
        pickle.dump(sampler_metrics, f)

# Run this to generate the metric matrix
#generate_undersampling_metric_matrix()

################################################ Creating DB2 ###############################################

# We know from the generated matrix that Instance Hardness Threshold Sampler with sampling strategy of 0.7 and cv = 3
# gives the best overall C.V accuracy

sampler = get_sampler("InstanceHardnessThreshold", {'cv': 3, 'sampling_strategy': 0.9})

DB2_X, DB2_y = sampler.fit_resample(D_input_cols, D_target)
print("\n\nDB2: Number of + class examples = ", np.count_nonzero(DB2_y))
print("\nDB2: Number of - class examples = ", (DB2_y.shape[0] - np.count_nonzero(DB2_y)))

acc= []
std = []

X_train, X_test, y_train, y_test = train_test_split(DB2_X,\
        DB2_y, test_size = 0.33, random_state = 5)

def run_search():
    hr()
    print("Generating metric matrix for different Classifiers on DB2")
    hr()
    best_params = {} 
    for i in range(len(classifiers)):
        # Using GridSearch to get the best combination of parameters
        classifier = GridSearchCV(classifiers[i](), classifier_param_dict[i], verbose=1, scoring="balanced_accuracy")
        classifier.fit(X_train, y_train)
        best_params[names[i]] = classifier.best_params_

    with open('pickled_objects/best_under_param_dict.pkl', 'wb') as f:
        pickle.dump(best_params, f)

# Run the below function to generate param metrics
run_search()

# Load the best parameters
with open('pickled_objects/best_under_param_dict.pkl', 'rb') as f:
    best_param = pickle.load(f)
    print("\n\nLoaded Best Params for Classifiers on DB2\n\n")
#################################### Retraining the 4 classifiers with DB2 ###################################
def retrain_on_DB2():
    hr()
    print("Training the Classifiers on DB2")
    hr()
    for i in range(len(classifiers)):
        name = names[i]
        classifier = classifiers[i](best_param[names[i]])
        met = run_classifier(classifier, X_train, y_train, X_test, y_test, split=False)
        cv_score = k_fold_cv(classifier, None, DB2_X, DB2_y, strat_cv, "balanced_accuracy")
        plot_confusion_matrices("Plots/algos_on_DB2", met, name, cv_score)
        plot_roc("Plots/algos_on_DB2", met["ROC"], name)

retrain_on_DB2()
#########################################################################################################
                            # TASK 5: Training MLP and Gradient Boosting
#########################################################################################################

# For MLP =>

from sklearn.neural_network import MLPClassifier

params = {
    "hidden_layer_sizes": [(512, 256, 64, 16, 4), (256, 128, 32, 8), (64, 32, 8)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.01, 0.1],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [2000]
    }

def run_MLP(X, y, dataset_name):
    print("Running MLP on Dataset: ", dataset_name)
    # hr()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.33)
    mlp = MLPClassifier()

    def get_param():
        classifier = GridSearchCV(mlp, params, verbose=2, scoring="balanced_accuracy")
        classifier.fit(X_train, y_train)
        best_param = classifier.best_params_
        print("\n\nBest Parameters for MLP on Dataset: ", dataset_name)
        print(best_param)

        with open(f'pickled_objects/MLP_best_param_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(best_param, f)

    get_param()

    with open(f'pickled_objects/MLP_best_param_{dataset_name}.pkl', 'rb') as f:
        best_param = pickle.load(f)

    classifier = MLPClassifier(**best_param)
    met = run_classifier(classifier, X_train, y_train, X_test, y_test, split=False)
    cv_score = k_fold_cv(classifier, None, X, y, strat_cv, "balanced_accuracy")
    plot_confusion_matrices("Plots/MLP", met, dataset_name, cv_score)
    plot_roc("Plots/MLP", met["ROC"], dataset_name)

# Uncomment the lines below to run MLP on Datasets D, DB1, and DB2
run_MLP(D_input_cols, D_target, "D")
run_MLP(DB1_X, DB1_y, "DB_1")
run_MLP(DB2_X, DB2_y, "DB_2")

# Gradient Boosting ==>
from sklearn.ensemble import GradientBoostingClassifier


params = {
    "loss": ["log_loss", "exponential"],
    'criterion': ['friedman_mse', 'squared_error'],
    'subsample': [0.1, 0.5, 1],
    'learning_rate': [0.0001, 0.01, 0.1],
    'n_estimators': [50, 100, 500],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.3, 0.5],
    'max_depth': [3, 5, 7],
    'max_features': ['sqrt', 'log2']
    }

def run_GB(X, y, dataset_name):
    print("Running GradientBoostingClassifier on Dataset: ", dataset_name)
    # hr()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.33)
    gbc = GradientBoostingClassifier()

    def get_param():    
        classifier = GridSearchCV(gbc, params, verbose=2, scoring="balanced_accuracy")
        classifier.fit(X_train, y_train)
        best_param = classifier.best_params_
        with open(f'pickled_objects/GB_best_param_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(best_param, f)

    get_param()

    with open(f'pickled_objects/GB_best_param_{dataset_name}.pkl', 'rb') as f:
        best_param = pickle.load(f)

    print("\n\nBest Parameters for GradientBoostingClassifier on Dataset: ", dataset_name)
    print(best_param)

    classifier = GradientBoostingClassifier(**best_param)
    met = run_classifier(classifier, X_train, y_train, X_test, y_test, split=False)
    cv_score = k_fold_cv(classifier, None, X, y, strat_cv, "balanced_accuracy")
    plot_confusion_matrices("Plots/GBC", met, dataset_name, cv_score)
    plot_roc("Plots/GBC", met["ROC"], dataset_name)

# Uncomment the lines below to run GradientBoostingClassifier on Datasets D, DB1, and DB2
run_GB(D_input_cols, D_target, "D")
run_GB(DB1_X, DB1_y, "DB_1")
run_GB(DB2_X, DB2_y, "DB_2")
