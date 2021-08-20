import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_roc_curve as plot_roc_curve_sk, classification_report

from matplotlib import pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

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


def plot_roc_curve(clfs, X, y, labels=None, shade=True,
    subplots_kwargs={}, save_fpath=None):
    fig, ax = plt.subplots(**subplots_kwargs)

    if isinstance(clfs, list):
        if labels is not None:
            assert isinstance(labels, list)
        for clf, label in zip(clfs, labels):
            displ = plot_roc_curve_sk(clf, X, y, ax=ax, name=label,
                ls="--", alpha=0.8)
            if shade:
                ax.fill_between(x=displ.fpr, y1=displ.tpr,
                    color=displ.line_.get_color(), alpha=0.1)
    else:
        plot_roc_curve_sk(clfs, X, y, ax=ax)

    xfit = np.linspace(0, 1, 1000)
    ax.plot(xfit, xfit, "--", label="Skill-Less Classifier (AUC=0.50)", alpha=0.8)
    plt.legend(fontsize="large", loc=4)

    if save_fpath is not None:
        plt.savefig(save_fpath, bbox_inches="tight")

    plt.show()

def print_classification_reports(clfs, Xs, ys, names):
    for clf, X, y, name in zip(clfs, Xs, ys, names):
        print(f"{name}: {sep}")
        print(classification_report(y, clf.predict(X)))


def red_greens_cmap():
    """
    Red-to-green seaborn cmap
    Ref:
    https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red
    """
    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0,.15,.4,.5,0.6,.9,1.]
    l = list(zip(v,c))
    return LinearSegmentedColormap.from_list('rg',l, N=256)


def classification_report_df(clf, X, y, clean=True,
    plot=True, save_fpath=None):
    df = pd.DataFrame(classification_report(y,clf.predict(X),
            output_dict=True))

    df.loc[:, "accuracy"] = [np.nan, np.nan, df.accuracy[0], df["macro avg"][-1]]
    df = df.T
    if clean:
        df = df.round(2)
        df.loc[:, "support"] = df.support.astype(int)

    if plot:
        sns.heatmap(df.drop("support", axis=1), cmap=red_greens_cmap(), annot=True)
        if save_fpath is not None:
            plt.savefig(save_fpath, bbox_inches="tight")
        plt.show()
    return df

def compare_classification_reports(clfs, Xs, ys, names, clean=True,
    plot=True, save_fpath=None, suptitle=None, subplot_kwargs={},
    cbar_orient="horizontal"):
    """
    Compare classification reports for classifiers
    Arguments:
    - clfs - list of classifiers
    - Xs - list of X dataframes (for each classifier)
    - ys - list of known classifications (for each classifier)
    - names - list of classifier names (for labelling)
    - clean - round all floats to 2.d.p.
    - plot - plot report comparison
    - save_fpath - file path to save plot
    - suptite - super title of figure
    - subplot_kwargs - kwargs for plt.suplots()
    - cbar_orient - orientation of cbar (either 'horizontal' or 'vertical')

    Returns:
    - dfs - list of classification report dataframes
    - plot - if plot==True, displays comparison plot
    """
    dfs = []
    for clf, X, y in zip(clfs, Xs, ys):
        dfs.append(classification_report_df(clf, X, y, clean=clean, plot=False))

    vmin = min([df.drop("support", axis=1).min().min() for df in dfs])
    vmax = max([df.drop("support", axis=1).max().max() for df in dfs])

    fig, axs = plt.subplots(**subplot_kwargs, sharex=True, sharey=True)
    cmap = red_greens_cmap()

    if plot:
        for df, ax, name in zip(dfs, axs, names):
            im = sns.heatmap(df.drop("support", axis=1),
                cmap=cmap, annot=True, vmin=vmin, vmax=vmax, cbar=False, ax=ax)
            ax.set_title(name)
            ax.tick_params(axis='y', rotation=0)

        mappable = im.get_children()[0]
        plt.colorbar(mappable, ax=axs, orientation = cbar_orient)

        if suptitle is not None:
            plt.suptitle(suptitle, fontsize="x-large")
        # fig.tight_layout()

        if save_fpath is not None:
            plt.savefig(save_fpath, bbox_inches="tight")

        plt.show()

    return dfs
