import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC


# 根据指纹组合方式获取特征
def get_finger(maccs, ecfp4, fcfp4, ls):
    length = len(ls)
    finger_name = ''
    if length == 3:
        x = pd.concat([maccs, ecfp4, fcfp4], axis=1)
        finger_name = 'maccs_ecfp4_fcfp4'
    elif length == 2:
        if sum(ls) == 5:
            x = pd.concat([ecfp4, fcfp4], axis=1)
            finger_name = 'ecfp4_fcfp4'
        elif sum(ls) == 4:
            x = pd.concat([maccs, fcfp4], axis=1)
            finger_name = 'maccs_fcfp4'
        else:
            x = pd.concat([maccs, ecfp4], axis=1)
            finger_name = 'maccs_ecfp4'
    else:
        if sum(ls) == 1:
            x = maccs
            finger_name = 'maccs'
        elif sum(ls) == 2:
            x = ecfp4
            finger_name = 'ecfp4'
        else:
            x = fcfp4
            finger_name = 'fcfp4'
    x.columns = [i for i in range(len(x.columns))]
    return x, finger_name


# 随机欠采样方法
def random_under_sampler(x, y):

    rus = RandomUnderSampler()
    x_train, y_train = rus.fit_resample(x, y)
    print('随机欠采样后训练集中正样本数：', np.sum(y_train == 1), '负样本数：', np.sum(y_train == 0), '总数：', len(x_train))

    # 获取未被采样的样本
    x_lost = x[~np.isin(np.arange(len(x)), y_train.index)]
    y_lost = y[~np.isin(np.arange(len(y)), y_train.index)]

    return x_train, y_train, x_lost, y_lost


# 混合采样
def combine_sampler(x, y):
    x_index = pd.DataFrame(columns=['axes', 'index'])
    x_index['axes'] = x.axes[0]
    x_index['index'] = [i for i in range(len(x_index))]
    # 使用SMOTETomek结合过采样和欠采样
    combined = SMOTETomek(random_state=42)

    # 应用混合采样
    x_resampled, y_resampled = combined.fit_resample(x_index, y)
    x_train = x.iloc[x_resampled['index']]

    return x_train, y_resampled


# 网格搜索寻找最优参数
def grid_search_method(model, param_grid, x_train, y_train):

    g_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='accuracy', verbose=1, n_jobs=-1)
    g_search.fit(x_train, y_train)
    return g_search


def get_best_parameters(set_type, finger_name, model_type):
    file_name = f'data/parameters/{set_type}/{finger_name}.txt'
    with open(file_name, 'r') as f:
        file = f.read()
        parameters = file.split('\n')
        parameters_ = parameters[model_type]
        parameters_ = parameters_[1:-1]
        parameter_ = parameters_.split(', ')
        parameter = {}
        for param in parameter_:
            key, value = param.split(': ')
            if value[0] == "'":
                parameter[key[1:-1]] = value[1:-1]
            elif value == 'True':
                parameter[key[1:-1]] = True
            elif value == 'False':
                parameter[key[1:-1]] = False
            elif value == 'None':
                parameter[key[1:-1]] = None
            elif len(value.split('.')) == 2:
                parameter[key[1:-1]] = float(value)
            else:
                parameter[key[1:-1]] = int(value)
    return parameter


# 模型性能评估
def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("AUC Score: ", auc_score)

    return accuracy, precision, recall, f1, auc_score


def write_record(root, message, addr, localtime):
    fw = open(f'{root}_{addr}_{localtime}.txt', 'a')
    fw.write(f'{message}\n')
    fw.close()


def write_result(root, message, data, addr, model, localtime):
    fw = f'{root}_{addr}_{model}_{localtime}.csv'
    data.to_csv(fw, index=False)
    write_record(root, message, addr, localtime)

