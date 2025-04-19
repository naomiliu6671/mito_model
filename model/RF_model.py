import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from process.process_main import evaluation, write_result, write_record, grid_search_method


def train_rf_model(x_train, y_train, root, localtime):
    print('Random Forest model start training...')
    # 定义参数网格
    param_grid = {
        'n_estimators': range(500, 1000, 100),  # range(500, 1000)
        'max_depth': [None, 5, 7, 9, 10],
        'min_samples_split': [5, 7, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
    }

    # 初始化随机森林分类器
    rf_model = RandomForestClassifier(random_state=99)
    print('Grid Search...')
    # 网格搜索最优参数
    grid_search = grid_search_method(rf_model, param_grid, x_train, y_train)
    print("RF Best parameters found: ", grid_search.best_params_)
    print("RF Best cross-validation score: ", grid_search.best_score_)
    write_record(root, f'{grid_search.best_params_}', 'train', localtime)

    return grid_search.best_estimator_


def predict_rf_model(x_test, best_estimator):
    print('Random Forest model start predicting...')
    # 预测
    y_pred = best_estimator.predict(x_test)
    return y_pred


def evaluate_rf_model(root, y_test, y_pred, localtime):
    print('Random Forest model start evaluating...')
    # 性能评估
    accuracy, precision, recall, f1, auc_score = evaluation(y_test, y_pred)
    data = pd.DataFrame([y_test, y_pred], columns=['y_test', 'y_pred'])
    write_result(root,
                 f'RF Result:\n pre.:{precision:.4f} rec.:{recall:.4f} f1:{f1:.4f} acc:{accuracy:.4f} auc:{auc_score:.4f}',
                 data, 'test', 'rf', localtime)


def save_rf_model(best_estimator, root, localtime):
    # 保存模型
    joblib.dump(best_estimator, f'{root}_RF_model_{localtime}.joblib')
    print('RF model saved successfully!')
