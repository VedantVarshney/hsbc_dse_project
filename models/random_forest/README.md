
    ## Adaboost models trained on SMOTE resampled data
    Contents:
    - `rf_base.p` - trained RandomForestClassifier instance (baseline classifier)
    - `rf_drop.p` - trained RandomForestClassifier instance (baseline classifier with dropped features)
    - `rf_best.p` - trained RandomForestClassifier instance (baseline classifier with dropped features and GridSearch)
    - `rf_smote.p` - trained RandomForestClassifier instance (baseline classifier with dropped features and SMOTE)
    - `data.p` - data tuple (X_train, X_drop, X_train_res_drop, X_test, X_test_drop, y_train, y_train_res_drop, y_test)
    - `low_importance_features.p` - pd Series of low importance features
    The following low importance features have been dropped:
    ['job_entrepreneur' 'job_housemaid' 'job_self-employed' 'job_student'
 'job_unemployed' 'job_unknown' 'default_yes' 'month_dec' 'poutcome_other']
    