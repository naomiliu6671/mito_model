import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from process.process_main import evaluation, write_record, write_result,grid_search_method


def train_lgb_model(x_train, y_train, root, localtime):
    print('LightGBM model start training...')
    # 定义参数网格
    param_grid = {
        'learning_rate': [0.05, 0.07, 0.09, 0.1],
        'n_estimators': [300, 500, 700, 800],
        'max_depth': range(1, 9, 2)
    }

    # 创建LightGBM分类器
    lgb_model = LGBMClassifier(objective='binary', random_state=42)
    print('Grid search...')
    # 网格搜索寻参
    grid_search = grid_search_method(lgb_model, param_grid, x_train, y_train)

    print('Best parameters found: ', grid_search.best_params_)
    print('Best cross-validation score: ', grid_search.best_score_)
    write_record(root, f'{grid_search.best_params_}', 'train', localtime)
    return grid_search.best_estimator_


def predict_lgb_model(x_test, best_estimator):
    print('LightGBM model start predicting...')
    # 使用最佳模型进行预测
    y_pred = best_estimator.predict(x_test)
    return y_pred


def evaluate_lgb_model(root, y_test, y_pred, localtime):
    print('LightGBM model start evaluating...')
    # 计算准确率
    accuracy, precision, recall, f1, auc_score = evaluation(y_test, y_pred)
    data = pd.DataFrame([y_test, y_pred], columns=['y_test', 'y_pred'])
    write_result(root,
                 f'LightGBM Result:\n pre.:{precision:.4f} rec.:{recall:.4f} f1:{f1:.4f} acc:{accuracy:.4f} auc:{auc_score:.4f}',
                 data, 'test', 'lightgbm', localtime)


def save_lgb_model(best_estimator, root, localtime):
    # 保存模型
    joblib.dump(best_estimator, f'{root}_lgb_model_{localtime}.joblib')
    print(f'LightGBM model saved successfully!')
