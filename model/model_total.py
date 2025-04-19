import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from process.process_main import get_best_parameters

from model.LightGBM import train_lgb_model, predict_lgb_model, evaluate_lgb_model, save_lgb_model
from model.SVM_model import train_svm_model, predict_svm_model, evaluate_svm_model, save_svm_model
from model.RF_model import train_rf_model, predict_rf_model, evaluate_rf_model, save_rf_model
from model.XGBoost_model import train_xgb_model, predict_xgb_model, evaluate_xgb_model, save_xgb_model
from model.soft_vote import soft_vote_model, predict_soft_vote_model, evaluate_soft_vote_model, save_soft_vote_model


def model_find_parameter(x_train, y_train, x_test, y_test, result_root):

    # 获取当前时间
    localtime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    # 在这里进行模型训练
    svm_model = train_svm_model(x_train, y_train, result_root, localtime)
    rf_model = train_rf_model(x_train, y_train, result_root, localtime)
    # xgboost_model = train_xgb_model(x_train, y_train, result_root, localtime)
    lightgbm_model = train_lgb_model(x_train, y_train, result_root, localtime)
    vote_model = soft_vote_model(svm_model, rf_model, lightgbm_model, x_train, y_train)

    vote_predict = predict_soft_vote_model(x_test, vote_model)
    evaluate_soft_vote_model(result_root, y_test, vote_predict, localtime)
    save_soft_vote_model(vote_model, result_root, localtime)
    print('find parameter end...')


def model_best_parameter(x_train, y_train, x_test, y_test, result_root, dataset_name, finger_name):

    # 获取当前时间
    localtime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())

    # 在这里进行模型训练
    print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())} SVM start training...')
    svm_param = get_best_parameters(dataset_name, finger_name, 0)
    svm_model = SVC(C=svm_param['C'], gamma=svm_param['gamma'], kernel=svm_param['kernel'], probability=True)
    # svm_model = SVC(probability=True)
    svm_model.fit(x_train, y_train)

    print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())} Random Forest start training...')
    rf_param = get_best_parameters(dataset_name, finger_name, 1)
    rf_model = RandomForestClassifier(bootstrap=rf_param['bootstrap'], max_depth=rf_param['max_depth'],
                                 max_features=rf_param['max_features'], min_samples_leaf=rf_param['min_samples_leaf'],
                                 min_samples_split=rf_param['min_samples_split'], n_estimators=rf_param['n_estimators'],
                                 random_state=99)
    # rf_model = RandomForestClassifier(random_state=99)
    rf_model.fit(x_train, y_train)

    # print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())} XGBoost start training...')
    # xgboost_param = get_best_parameters(dataset_name, finger_name, 2)
    # xgboost_model = XGBClassifier(gamma=xgboost_param['gamma'], max_depth=xgboost_param['max_depth'],
    #                               min_child_weight=xgboost_param['min_child_weight'],
    #                               colsample_bytree=xgboost_param['colsample_bytree'],
    #                               subsample=xgboost_param['subsample'],
    #                               n_estimators=xgboost_param['n_estimators'],
    #                               random_state=25, objective='binary:logistic')
    # xgboost_model = XGBClassifier(random_state=25, objective='binary:logistic')
    # xgboost_model.fit(x_train, y_train)

    print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())} LightGBM start training...')
    lightgbm_param = get_best_parameters(dataset_name, finger_name, 3)
    lightgbm_model = LGBMClassifier(max_depth=lightgbm_param['max_depth'],
                                    learning_rate=lightgbm_param['learning_rate'],
                                    n_estimators=lightgbm_param['n_estimators'], objective='binary', random_state=42)
    # lightgbm_model = LGBMClassifier(objective='binary', random_state=42)
    lightgbm_model.fit(x_train, y_train)

    print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())} Vote start training...')
    # vote_model = soft_vote_model(svm_model, rf_model, xgboost_model, lightgbm_model, x_train, y_train)
    vote_model = soft_vote_model(svm_model, rf_model, lightgbm_model, x_train, y_train)

    # 在这里进行模型预测
    y_svm_pred = predict_svm_model(x_test, svm_model)
    y_rf_pred = predict_rf_model(x_test, rf_model)
    # y_xgboost_pred = predict_xgb_model(x_test, xgboost_model)
    y_lightgbm_pred = predict_lgb_model(x_test, lightgbm_model)
    y_soft_vote_pred = predict_soft_vote_model(x_test, vote_model)

    # 在这里进行模型评估
    evaluate_svm_model(result_root, y_test, y_svm_pred, localtime)
    evaluate_rf_model(result_root, y_test, y_rf_pred, localtime)
    # evaluate_xgb_model(result_root, y_test, y_xgboost_pred, localtime)
    evaluate_lgb_model(result_root, y_test, y_lightgbm_pred, localtime)
    evaluate_soft_vote_model(result_root, y_test, y_soft_vote_pred, localtime)

    # 在这里保存最优模型
    save_soft_vote_model(vote_model, result_root, localtime)


def model_use(smiles):
    model = joblib.load('outputs/model/Soft_vote_model_final.joblib')

    tox = model.predict(smiles)
    mito_tox = pd.DataFrame([smiles, tox], columns=['SMILES', 'Tox'])
    return mito_tox

