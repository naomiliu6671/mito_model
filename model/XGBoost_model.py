import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from process.process_main import evaluation, write_record, write_result, grid_search_method


def train_xgb_model(x_train, y_train, root, localtime):
    print("XGBoost model start training...")
    parameters = {
        'n_estimators': np.arange(6, 11, 2),
        'max_depth': np.arange(6, 11, 2),
        'colsample_bytree': np.arange(0.8, 0.9, 0.05),
        'subsample': np.arange(0.8, 0.9, 0.05),
        'gamma': np.arange(0.1, 0.5, 0.2),
        'min_child_weight': np.arange(1, 3, 1)
    }

    xgboost_model = xgb.XGBClassifier(random_state=25, objective='binary:logistic')
    print('Grid search...')
    grid_search = grid_search_method(xgboost_model, parameters, x_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    write_record(root, f'{grid_search.best_params_}', 'train', localtime)

    return grid_search.best_estimator_


def predict_xgb_model(x_test, best_estimator):
    print("XGBoost model start predicting...")
    # 使用最佳模型进行预测
    y_pred = best_estimator.predict(x_test)
    return y_pred


def evaluate_xgb_model(root, y_test, y_pred, localtime):
    print("XGBoost model start evaluating...")
    # 计算准确率
    accuracy, precision, recall, f1, auc_score = evaluation(y_test, y_pred)
    data = pd.DataFrame([y_test, y_pred], columns=['y_test', 'y_pred'])
    write_result(root,
                 f'XGBoost Result:\n pre.:{precision:.4f} rec.:{recall:.4f} f1:{f1:.4f} acc:{accuracy:.4f} auc:{auc_score:.4f}',
                 data, 'test', 'xgboost', localtime)


def save_xgb_model(best_estimator, root, localtime):
    # 保存模型
    joblib.dump(best_estimator, f'{root}_xgb_{localtime}.joblib')
    print(f'XGBoost model saved successfully!')
