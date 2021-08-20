
    ## Adaboost models trained on SMOTE resampled data
    Contents:
    - `dt_base.p` - trained DecisionTreeClassifier instance (baseline classifier)
    - `dt_drop.p` - trained DecisionTreeClassifier instance (baseline classifier with dropped features)
    - `dt_best.p` - trained DecisionTreeClassifier instance (best classifier)
    - `dt_drop_res.p` - trained DecisionTreeClassifier instance (baseline classifier with dropped features and SMOTE)
    - `data.p` - data tuple (X_train, X_train_drop, X_train_res_drop, X_test, X_test_drop, y_train, y_train_res_drop, y_test)
    - `low_importance_features.p` - pd Series of low importance features
    The following low importance features have been dropped:
    ['job_entrepreneur' 'job_housemaid' 'job_retired' 'job_self-employed'
 'job_services' 'job_student' 'job_unemployed' 'job_unknown'
 'education_unknown' 'default_yes' 'contact_telephone' 'month_dec'
 'month_jan' 'poutcome_other' 'poutcome_unknown']
    