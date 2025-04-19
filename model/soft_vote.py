import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from process.process_main import evaluation, write_result


def soft_vote_model(svm_model, rf_model, lightgbm_model, x_train, y_train):
    print('Soft-Vote model training start...')
    # 集成学习-投票法
    # soft_vote = VotingClassifier(
    #     estimators=[('svm', svm_model), ('rf', rf_model), ('xgb', xgboost_model), ('lgb', lightgbm_modle)],
    #     voting='soft')
    soft_vote = VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model), ('lgb', lightgbm_model)],
        voting='soft')
    soft_vote.fit(x_train, y_train)
    return soft_vote


def predict_soft_vote_model(x_test, model):
    print('Soft-Vote model start predicting...')
    # 使用最佳模型进行预测
    y_pred = model.predict(x_test)
    return y_pred


def evaluate_soft_vote_model(root, y_test, y_pred, localtime):
    print('Soft-Vote model start evaluating...')
    # 模型评估
    accuracy, precision, recall, f1, auc_score = evaluation(y_test, y_pred)
    data = pd.DataFrame([y_test, y_pred], columns=['y_test', 'y_pred'])
    write_result(root,
                 f'Soft-Vote Result:\n pre.:{precision:.4f} rec.:{recall:.4f} f1:{f1:.4f} acc:{accuracy:.4f} auc:{auc_score:.4f}',
                 data, 'test', 'soft_vote', localtime)


def save_soft_vote_model(best_estimator, root, localtime):
    # 保存模型
    joblib.dump(best_estimator, f'{root}_Soft_Vote_model_{localtime}.joblib')
    print(f'Soft-Vote model saved successfully!')
