from sklearn.feature_selection import VarianceThreshold

def select_features(variables, mode="FirstSix"):
    if mode == "FirstSix":
        return variables.iloc[:, :6]
    elif mode == "VarianceThreshold":
        selector = VarianceThreshold(threshold=(0.9 * (1 - 0.9)))
        return selector.fit_transform(variables)
