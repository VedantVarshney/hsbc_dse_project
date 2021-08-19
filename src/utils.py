import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_roc_curve as plot_roc_curve_sk, classification_report

from matplotlib import pyplot as plt

# Misc variables (useful for formatting print reports)
sep = "\n_________________________________________\n"

def preprocess(df,
              cont_cols = ["age", "balance", "duration"],
              cont_transform = lambda x: x,
              copy=True,
              test_size=0.25,
              target="y",
              drop_cols=None,
              random_state=None):
    """
    Preprocesses the dataset by applying transforms and adding dummy
    variables for categorical features.
    Function assumes the target variable is a binary variable of 'yes' and 'no'

    Arguments:
    - df - data - Pandas dataframe
    - cont_cols - column names for continous features
    - cont_transform - transformation function for continuous features
    - copy - copy dataframe when applying changes

    Returns:
    - X_train, X_test, y_train, y_test
    - scaler - fitted StandardScaler for the *transformed* continuous features
    """
    assert set(df[target].unique()) == set(["yes", "no"])
    assert isfinite_df(df), "Null values found in dataframe"

    if copy:
        df = df.copy()

    if drop_cols is not None:
        df.drop(drop_cols, inplace=True)

    y = df[target].apply(lambda x: x=="yes").astype(int)
    X = df.drop(target, axis=1)
    # Unused
    del df

#     Transform continous columns
    transf_cols = [f"{col}_transf" for col in cont_cols]
    X[transf_cols] = X[cont_cols].apply(cont_transform)
    X.drop(cont_cols, axis=1, inplace=True)

    scaler = StandardScaler()
    X.loc[:, transf_cols] = scaler.fit_transform(X[transf_cols])

    X = pd.get_dummies(X, drop_first=True)

    # print(pd.isnull(X).any())
    # print(X["balance_transf"])
    assert isfinite_df(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state)

    for data in [X_train, X_test]:
        assert isfinite_df(data), "Null values found in dataframe"

    for data in [y_train, y_test]:
        assert isfinite_series(data), "Null values found in series"

    return X_train, X_test, y_train, y_test, scaler


def isfinite_df(X):
    """
    Returns True if no value in a pd dataframe is null
    """
    return not pd.isnull(X).any().any()

def isfinite_series(x):
    """
    Returns True if no value in a pd series is null
    """
    return not pd.isnull(x).any()


def plot_feature_importances(feature_importances, feature_names,
    thresh=None, return_low_importance_features=False):

    feature_importances = pd.Series(feature_importances, index=feature_names,
        name="Feature Importances")

    fig, ax = plt.subplots(figsize=(8, 12))
    feature_importances.plot(kind="barh", ax=ax)

    if thresh is not None:
        ax.axvline(thresh, color="r", ls="--")

    ax.set_title("Feature Importances")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Importance (arb. unit)")

    fig.tight_layout()
    plt.show()

    if return_low_importance_features:
        assert isinstance(thresh, (float, int))
        return feature_importances[feature_importances < thresh]


def plot_roc_curve(clfs, X, y):
    fig, ax = plt.subplots()

    if isinstance(clfs, list):
        for clf in clfs:
            plot_roc_curve_sk(clf, X, y, ax=ax)
    else:
        plot_roc_curve_sk(clfs, X, y, ax=ax)

    xfit = np.linspace(0, 1, 1000)
    ax.plot(xfit, xfit, "--", label="Skill-Less Classifier")

    plt.legend()
    plt.show()

def print_classification_reports(clfs, Xs, ys, names):
    for clf, X, y, name in zip(clfs, Xs, ys, names):
        print(f"{name}: {sep}")
        print(classification_report(y, clf.predict(X)))
