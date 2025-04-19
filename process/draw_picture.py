import matplotlib.pyplot as plt
import numpy as np


evaluation_type = {'Acc': 231, 'Pre': 232, 'Recall': 233, 'F1': 234, 'AUC': 235}

def draw_subplot(x, y, x_label, y_label, title, save_path):
    plt.subplot(evaluation_type[title])
    plt.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def draw_evaluation_picture(data):
    for finger in evaluation_type.keys():
        sub_data = data[['fingerprint', finger, 'method']]

