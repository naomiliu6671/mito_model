import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from process.process_main import random_under_sampler, get_finger, combine_sampler
from model.model_total import model_find_parameter, model_best_parameter


def main():
    data_root = 'data'
    result_root = 'outputs'
    times = 5
    file = pd.read_csv(f'{data_root}/mito_data.csv')

    x_smiles = file
    y = file['type']

    # 划分原始训练集和测试集
    x_train0, x_test0, y_train0, y_test0 = train_test_split(x_smiles, y, test_size=0.3, random_state=42)
    print(f'原始训练集大小为{x_train0.shape[0]}，测试集大小为{x_test0.shape[0]}')
    print('训练集中原始正样本数：', np.sum(y_train0 == 1), '原始负样本数：', np.sum(y_train0 == 0))

    # 随机欠采样-多余数据加入测试集
    x_train1, y_train1, x_lost, y_lost = random_under_sampler(x_train0, y_train0)
    x_test1, y_test1 = x_test0, y_test0
    x_test1 = x_test1.append(x_lost)
    y_test1 = y_test1.append(y_lost)
    print(f'随机欠采样后，训练集大小为{x_train1.shape[0]}，测试集大小为{x_test1.shape[0]}')

    # 混合采样
    x_train2, y_train2 = combine_sampler(x_train0, y_train0)
    x_test2, y_test2 = x_test0, y_test0
    print(f'混合采样后训练集大小为{x_train2.shape[0]}，测试集大小为{x_test2.shape[0]}')

    # 获取指纹
    maccs = pd.read_csv(f'{data_root}/mito_maccs.csv')
    ecfp4 = pd.read_csv(f'{data_root}/mito_ecfp4.csv')
    fcfp4 = pd.read_csv(f'{data_root}/mito_fcfp4.csv')

    # 4种训练集和测试集搭配方法
    dataset = ['raw', 'random_under_sampling', 'random_under_sampling_test0', 'combine_sampling']

    # 指纹组合
    finger_ls = [[1, 3]]     # [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]

    for i in finger_ls:
        x_train0_ = x_train0['smiles']
        x_test0_ = x_test0['smiles']
        x_train1_ = x_train1['smiles']
        x_test1_ = x_test1['smiles']
        x_train2_ = x_train2['smiles']
        x_test2_ = x_test2['smiles']

        maccs_, ecfp4_, fcfp4_ = maccs, ecfp4, fcfp4
        smiles_finger, finger_name = get_finger(maccs_, ecfp4_, fcfp4_, i)
        smiles_finger_ = pd.concat([smiles_finger, x_smiles['smiles']], axis=1)
        print(f'本次选择的指纹包括：{finger_name}')

        # 得到原始划分数据集对应的特征
        # x_train0_ = pd.merge(x_train0_, smiles_finger_, on='smiles', how='left')
        # x_train0_ = x_train0_.drop(['smiles'], axis=1)
        x_test0_ = pd.merge(x_test0_, smiles_finger_, on='smiles', how='left')
        x_test0_ = x_test0_.drop(['smiles'], axis=1)
        # print(f'融合指纹后得到原始训练集大小为{x_train0_.shape}，测试集大小为{x_test0_.shape}')
        # print('训练集中原始正样本数：', np.sum(y_train0 == 1), '原始负样本数：', np.sum(y_train0 == 0))

        # 模型
        # model_find_parameter(x_train0_, y_train0, x_test0_, y_test0,
        #                      f'{result_root}/find_parameters/{dataset[0]}/{finger_name}')
        # model_best_parameter(x_train0_, y_train0, x_test0_, y_test0,
        #                      f'{result_root}/{times}/{dataset[0]}/{finger_name}', dataset[0], finger_name)

        # 随机欠采样
        x_train1_ = pd.merge(x_train1_, smiles_finger_, on='smiles', how='left')
        x_train1_ = x_train1_.drop(['smiles'], axis=1)
        x_test1_ = pd.merge(x_test1_, smiles_finger_, on='smiles', how='left')
        x_test1_ = x_test1_.drop(['smiles'], axis=1)
        print(f'融合指纹后得到随机欠采样训练集大小为{x_train1_.shape}，测试集大小为{x_test1_.shape}')
        print('训练集中随机欠采样正样本数：', np.sum(y_train1 == 1), '随机欠采样负样本数：', np.sum(y_train1 == 0))

        # 模型
        # model_find_parameter(x_train1_, y_train1, x_test1_, y_test1,
        #                      f'{result_root}/find_parameters/{dataset[1]}/{finger_name}')
        model_best_parameter(x_train1_, y_train1, x_test1_, y_test1,
                             f'{result_root}/{times}/{dataset[1]}/{finger_name}', dataset[1], finger_name)


        # 随机欠采样——不改变测试集
        print(f'融合指纹后不改变测试集大小得到随机欠采样训练集大小为{x_train1_.shape}，测试集大小为{x_test0_.shape}')
        print('训练集中随机欠采样正样本数：', np.sum(y_train1 == 1), '随机欠采样负样本数：', np.sum(y_train1 == 0))
        # model_find_parameter(x_train1_, y_train1, x_test0_, y_test0,
        #                      f'{result_root}/find_parameters/{dataset[2]}/{finger_name}')
        model_best_parameter(x_train1_, y_train1, x_test0_, y_test0,
                             f'{result_root}/{times}/{dataset[2]}/{finger_name}', dataset[2], finger_name)


        # 混合采样
        # x_train2_ = pd.merge(x_train2_, smiles_finger_, on='smiles', how='left')
        # x_train2_ = x_train2_.drop(['smiles'], axis=1)
        # x_test2_ = pd.merge(x_test2_, smiles_finger_, on='smiles', how='left')
        # x_test2_ = x_test2_.drop(['smiles'], axis=1)
        # print(f'融合指纹后得到混合采样训练集大小为{x_train2_.shape}，测试集大小为{x_test2_.shape}')
        # print('训练集中混合采样正样本数：', np.sum(y_train2 == 1), '混合采样负样本数：', np.sum(y_train2 == 0))

        # 模型
        # model_find_parameter(x_train2_, y_train2, x_test2_, y_test2,
        #                      f'{result_root}/find_parameters/{dataset[3]}/{finger_name}')
        # model_best_parameter(x_train2_, y_train2, x_test2_, y_test2,
        #                      f'{result_root}/{times}/{dataset[3]}/{finger_name}', dataset[3], finger_name)


if __name__ == '__main__':
    main()

