import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve

import numpy as np

import gc

# train/test data

df_train_1= pd.read_csv("/Users/chenchingchun/Desktop/kebuke/answer_set/20181218_train1.csv")

df_test_1= pd.read_csv('/Users/chenchingchun/Desktop/kebuke/answer_set/20181218_test1.csv')

# First predict by lightgbm

#shuffule data
##try 111 222 333 444 555 666 777 888 999 101010
folds = StratifiedKFold(n_splits= 10, shuffle=True, random_state=444)

# Create arrays and dataframes to store results

oof_preds = np.zeros(df_train_1.shape[0])

sub_preds = np.zeros(df_test_1.shape[0])

feature_importance_df = pd.DataFrame()



feats = [f for f in df_train_1.columns if f not in ['order_id','deal_or_not','group_id',"Unnamed: 0",
                                                    "src_dst.去程",
                                                    "src_dst.回程",
                                                    "product_name",
                                                    'Begin_Date'
                                                    ]]


for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train_1[feats], df_train_1['deal_or_not'])):

    train_x, train_y = df_train_1[feats].iloc[train_idx], df_train_1['deal_or_not'].iloc[train_idx]

    valid_x, valid_y = df_train_1[feats].iloc[valid_idx], df_train_1['deal_or_not'].iloc[valid_idx]

sum(train_y == 0)/sum(train_y == 1)

train_x.info()

params = {
'nthread': 32,

'boosting_type': 'gbdt', #dart #gbdt #goss

'objective': 'binary',

'scale_pos_weight': sum(train_y == 0)/sum(train_y == 1),

'metric': 'auc',

'learning_rate': 0.01,

'num_leaves': 70,

'max_bin': 500,

'max_depth': 9,

'subsample': 1,

'feature_fraction': 0.8,

'colsample_bytree': 0.08,

'min_split_gain': 0.09,

'min_child_weight': 9.5,

'min_data_in_leaf': 150,

#'reg_alpha': 1,

#'reg_lambda': 50,

'verbose': 1,

# parameters for dart

'drop_rate':0.7,

'skip_drop':0.7,

'max_drop':5,

'uniform_drop':False,

'xgboost_dart_mode':True,

'drop_seed':4

}

if n_fold >= 0:

    dtrain = lgb.Dataset(

    train_x, label=train_y
    #, categorical_feature=["Source_1","Source_2",'SubLine']
    )

    dval = lgb.Dataset(

    valid_x, label=valid_y, reference=dtrain
    #, categorical_feature=["Source_1","Source_2","SubLine"]
    )
#111 0.749844 750472
bst1 = lgb.train(
    params, dtrain, num_boost_round=15000,
    valid_sets=[dval], early_stopping_rounds=500, verbose_eval=100)
#222 0.74148 742003
bst2 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#333 0.747047 746928
bst3 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#444 0.748967
bst4 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#555 0.746054
bst5 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#666 0.741908
bst6 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#777 0.745272
bst7 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#888 0.739663
bst8 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#999 0.742645
bst9 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)
#101010 0.745115
bst10 = lgb.train(
    params, dtrain, num_boost_round=10000,
    valid_sets=[dval], early_stopping_rounds=600, verbose_eval=100)


param_grid = {
#'nthread': 32,
'boosting_type': ['dart'], #dart #gbdt #goss

'objective': ['binary'],

'scale_pos_weight':[1.5,2,2.5,3],

'metric': ['auc'],

'learning_rate': [0.3],

'num_leaves': [20,30,35],

'max_bin':[800],

'max_depth': [9],

'subsample': [1],

'feature_fraction': [0.8,0.9],

'colsample_bytree': [0.08],

'min_split_gain': [0.09],

'min_child_weight': [9.5],

'min_data_in_leaf': [100],

"importance_type" :['gain'],

#'reg_alpha': 1,

#'reg_lambda': 50,

'verbose': [1],

# parameters for dart

'drop_rate':[0.7],

'skip_drop':[0.7],

'max_drop':[5],

'uniform_drop':[False],

'xgboost_dart_mode':[True],

'drop_seed':[4]
}

from sklearn.model_selection import GridSearchCV
mdl = LGBMClassifier()
model = GridSearchCV(estimator=mdl, param_grid=param_grid, n_jobs=32, cv=5, verbose=20,scoring='roc_auc')
model.fit(df_train_1[feats], df_train_1["deal_or_not"])
print(model.best_params_)
print(model.best_score_)

# Make the feature importance dataframe

gain1 = bst1.feature_importance('gain')

fold_importance_df = pd.DataFrame({'feature':bst1.feature_name(),

'split':bst1.feature_importance('split'),

'gain' :gain1,

'gain_%':100*gain1/gain1.sum(),

'fold':n_fold,

}).sort_values('gain',ascending=False)

# bagging
tmp1   = bst1.predict(df_test_1[feats], num_iteration=bst1.best_iteration)
tmp2   = bst2.predict(df_test_1[feats], num_iteration=bst2.best_iteration)
tmp3   = bst3.predict(df_test_1[feats], num_iteration=bst3.best_iteration)
tmp4   = bst4.predict(df_test_1[feats], num_iteration=bst4.best_iteration)
tmp5   = bst5.predict(df_test_1[feats], num_iteration=bst5.best_iteration)
tmp6   = bst6.predict(df_test_1[feats], num_iteration=bst6.best_iteration)
tmp7   = bst7.predict(df_test_1[feats], num_iteration=bst7.best_iteration)
tmp8   = bst8.predict(df_test_1[feats], num_iteration=bst8.best_iteration)
tmp9   = bst9.predict(df_test_1[feats], num_iteration=bst9.best_iteration)
tmp10  = bst10.predict(df_test_1[feats], num_iteration=bst10.best_iteration)
sub_preds = (tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8 + tmp9 + tmp10)/ 10

# create output sub-folder
import os
print(os.getcwd()) # Prints the current working directory
os.chdir('/Users/chenchingchun/Desktop/kebuke/answer_set')  # Provide the new path here

preds.to_csv("python_test2.csv", index=False)