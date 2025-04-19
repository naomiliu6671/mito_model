import joblib
import pandas as pd
from sklearn.svm import SVC
from process.process_main import evaluation, write_result, write_record, grid_search_method


def train_svm_model(x_train, y_train, root, localtime):
    print('SVM model start training...')
    # 定义支持向量机模型
    svm_model = SVC(probability=True)
    # 定义参数网格
    param_grid = {
        'C': [0.01, 0.1, 0.5, 1, 2, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }

    print('Grid search...')
    # 使用网格搜索找到最佳参数
    grid_search = grid_search_method(svm_model, param_grid, x_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    write_record(root, f'{grid_search.best_params_}', 'train', localtime)

    return grid_search.best_estimator_


def predict_svm_model(x_test, best_estimator):
    print('SVM model start predicting...')
    # 使用最佳模型进行预测
    y_pred = best_estimator.predict(x_test)
    return y_pred


def evaluate_svm_model(root, y_test, y_pred, localtime):
    print('SVM model start evaluating...')
    # 计算准确率
    accuracy, precision, recall, f1, auc_score = evaluation(y_test, y_pred)
    data = pd.DataFrame([y_test, y_pred], columns=['y_test', 'y_pred'])
    write_result(root,
                 f'SVM Result:\n pre.:{precision:.4f} rec.:{recall:.4f} f1:{f1:.4f} acc:{accuracy:.4f} auc:{auc_score:.4f}',
                 data, 'test', 'svm', localtime)


def save_svm_model(best_estimator, root, localtime):
    # 保存模型
    joblib.dump(best_estimator, f'{root}_SVM_model_{localtime}.joblib')
    print("SVM Model saved successfully!")
