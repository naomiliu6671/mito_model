import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def draw_result_picture(metrics, result_dict, fingerprints, methods):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    colors = ['#A6CEE3', '#AADCA9', '#BCF4C5', '#FCECCA', '#FCC3B4', '#FD9BA0', '#F07874']
    width = 0.2
    fig, ax = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
    axs = ax.flatten()
    number = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, metric in enumerate(metrics):
        x = np.arange(len(methods)) * 1.6
        ax = axs[i]
        min_value = 1
        max_value = 0
        for j, finger in enumerate(fingerprints):
            values = [result_dict[metric][finger][method] for method in methods]
            min_value = min(min_value, min(values))
            max_value = max(max_value, max(values))
            ax.bar(x + j * width, values, width, label=finger, color=colors[j])
        bottom = min(min_value - 0.4, 0.4)
        top_value = min(max_value + 0.1, 1)
        ax.set_ylim(bottom=bottom, top=top_value)
        ax.set_ylabel('Scores', fontsize=23, fontweight='bold')
        ax.set_xlabel('Methods', fontsize=23, fontweight='bold')
        title_ = '(' + number[i] + ') ' + metric + ' of Different Methods'
        ax.set_title(title_, fontweight='bold', fontsize=23, y=1.01)
        ax.set_xticks(x + width * (len(fingerprints) - 1) / 2)
        ax.set_xticklabels(methods, fontsize=20, fontweight='bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(20)

    axs[5].axis('off')
    handles, labels = ax.get_legend_handles_labels()  # 使用最后一个有效子图的handles和labels
    # axs[5].legend(handles, labels, loc='center', ncol=1, fontsize=18, title='Fingerprints', title_fontsize=20, )
    legend = axs[5].legend(handles, labels, loc='center', ncol=1, title="Fingerprints",
                            frameon=False, handlelength=1.5, handleheight=1, handletextpad=0.5, borderpad=1,
                            title_fontsize=22, labelspacing=1, prop={'size': 22, 'weight': 'bold'})
    legend.get_title().set_fontweight('bold')

    # 微调图例标题的位置，使其与图例项对齐
    for text in legend.get_texts():
        text.set_fontname('DejaVu Serif')  # 设置字体为所选字体
    legend._legend_box.align = "left"  # 尝试左对齐图例标题和图例项

    plt.tight_layout()
    plt.savefig('mito_result1.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.show()


result = pd.read_csv('data/result.csv')
methods_ = result['method'].unique()
fingerprints_ = result['fingerprint'].unique()
metrics_ = result.columns[1:-1]
result_dict_ = {metric: {fingerprint: {method: [] for method in methods_} for fingerprint in fingerprints_} for metric in metrics_}
for metric in metrics_:
    for method in methods_:
        for fingerprint in fingerprints_:
            value = result[(result['method'] == method) & (result['fingerprint'] == fingerprint)][metric].values
            result_dict_[metric][fingerprint][method] = value[0]
draw_result_picture(metrics_, result_dict_, fingerprints_, methods_)
