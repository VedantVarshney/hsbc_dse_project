
    ## Adaboost models trained on SMOTE resampled data
    Contents:
    - `adaboost_base.p` - trained AdaBoostClassifier instance (baseline classifier)
    - `adaboost_best.p` - trained AdaBoostClassifier instance (best classifier)
    - `data.p` - data tuple (X_train_res_drop, X_test_drop, y_train_res, y_test)
    - `low_importance_features.p` - pd Series of low importance features
    The following low importance features have been dropped:
    ['day' 'previous' 'default_yes' 'month_mar' 'month_oct' 'month_sep'
 'poutcome_other' 'poutcome_unknown']
    