import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
import pydot

import numpy as np
import copy
import glob
import io
import os
import sys
import time
import logging

Mws_obs_train = np.load('./scardec_usgs_obs/obs_train_Mws.npy')
Mws_obs_valid = np.load('./scardec_usgs_obs/obs_valid_Mws.npy')
Mws_obs = np.concatenate((Mws_obs_train, Mws_obs_valid))
Mws_obs_test = np.load('./scardec_usgs_obs/obs_test_Mws.npy')

Mws_syn_train = np.load('./syn_Mw_npy_test/syn_train_Mws.npy')
Mws_syn_valid = np.load('./syn_Mw_npy_test/syn_valid_Mws.npy')
Mws_syn = np.concatenate((Mws_syn_train, Mws_syn_valid))
Mws_syn_test = np.load('./syn_Mw_npy_test/syn_test_Mws.npy')

error_20_Mws_obs_test = np.load('./20%_obs_npy/predicted_20%_test_obs_usgs_errors.npy')
error_20_Mws_obs_train = np.load('./20%_obs_npy/predicted_20%_train_obs_usgs_errors.npy')
error_20_Mws_syn_test = np.load('./20%_syn_npy/predicted_20%_test_syn_errors.npy')
error_20_Mws_syn_train = np.load('./20%_syn_npy/predicted_20%_train_syn_errors.npy')

p_20_Mws_obs_test = np.load('./20%_obs_npy/predicted_20%_test_obs_usgs.npy')
p_20_Mws_obs_train = np.load('./20%_obs_npy/predicted_20%_train_obs_usgs.npy')
p_20_Mws_syn_test = np.load('./20%_syn_npy/predicted_20%_test_syn.npy')
p_20_Mws_syn_train = np.load('./20%_syn_npy/predicted_20%_train_syn.npy')

mask1 = (Mws_obs >= 6.0)
Mws_obs_above6 = Mws_obs[mask1]
indices_obs_Mws_above6 = np.where(mask1)[0]
p_20_Mws_obs_train_above6 = p_20_Mws_obs_train[indices_obs_Mws_above6]
error_20_Mws_obs_train_above6 = error_20_Mws_obs_train[indices_obs_Mws_above6]
mask2 = (Mws_obs_test >= 6.0)
Mws_obs_test_above6 = Mws_obs_test[mask2]
indices_obs_Mws_test_above6 = np.where(mask2)[0]
p_20_Mws_obs_test_above6 = p_20_Mws_obs_test[indices_obs_Mws_test_above6]
error_20_Mws_obs_test_above6 = error_20_Mws_obs_test[indices_obs_Mws_test_above6]

error_10_Mws_obs_test = np.load('./10%_obs_npy/predicted_10%_test_obs_usgs_errors.npy')
error_10_Mws_obs_train = np.load('./10%_obs_npy/predicted_10%_train_obs_usgs_errors.npy')
error_10_Mws_syn_test = np.load('./10%_syn_npy/predicted_10%_test_syn_errors.npy')
error_10_Mws_syn_train = np.load('./10%_syn_npy/predicted_10%_train_syn_errors.npy')

p_10_Mws_obs_test = np.load('./10%_obs_npy/predicted_10%_test_obs_usgs.npy')
p_10_Mws_obs_train = np.load('./10%_obs_npy/predicted_10%_train_obs_usgs.npy')
p_10_Mws_syn_test = np.load('./10%_syn_npy/predicted_10%_test_syn.npy')
p_10_Mws_syn_train = np.load('./10%_syn_npy/predicted_10%_train_syn.npy')

p_10_Mws_obs_train_above6 = p_10_Mws_obs_train[indices_obs_Mws_above6]
error_10_Mws_obs_train_above6 = error_10_Mws_obs_train[indices_obs_Mws_above6]
p_10_Mws_obs_test_above6 = p_10_Mws_obs_test[indices_obs_Mws_test_above6]
error_10_Mws_obs_test_above6 = error_10_Mws_obs_test[indices_obs_Mws_test_above6]

error_4_Mws_obs_test = np.load('./4%_obs_npy/predicted_4%_test_obs_usgs_errors.npy')
error_4_Mws_obs_train = np.load('./4%_obs_npy/predicted_4%_train_obs_usgs_errors.npy')
error_4_Mws_syn_test = np.load('./4%_syn_npy/predicted_4%_test_syn_errors.npy')
error_4_Mws_syn_train = np.load('./4%_syn_npy/predicted_4%_train_syn_errors.npy')

p_4_Mws_obs_test = np.load('./4%_obs_npy/predicted_4%_test_obs_usgs.npy')
p_4_Mws_obs_train = np.load('./4%_obs_npy/predicted_4%_train_obs_usgs.npy')
p_4_Mws_syn_test = np.load('./4%_syn_npy/predicted_4%_test_syn.npy')
p_4_Mws_syn_train = np.load('./4%_syn_npy/predicted_4%_train_syn.npy')

p_4_Mws_obs_train_above6 = p_4_Mws_obs_train[indices_obs_Mws_above6]
error_4_Mws_obs_train_above6 = error_4_Mws_obs_train[indices_obs_Mws_above6]
p_4_Mws_obs_test_above6 = p_4_Mws_obs_test[indices_obs_Mws_test_above6]
error_4_Mws_obs_test_above6 = error_4_Mws_obs_test[indices_obs_Mws_test_above6]

error_2_Mws_obs_test = np.load('./2%_obs_npy/predicted_2%_test_obs_usgs_errors.npy')
error_2_Mws_obs_train = np.load('./2%_obs_npy/predicted_2%_train_obs_usgs_errors.npy')
error_2_Mws_syn_test = np.load('./2%_syn_npy/predicted_2%_test_syn_errors.npy')
error_2_Mws_syn_train = np.load('./2%_syn_npy/predicted_2%_train_syn_errors.npy')

p_2_Mws_obs_test = np.load('./2%_obs_npy/predicted_2%_test_obs_usgs.npy')
p_2_Mws_obs_train = np.load('./2%_obs_npy/predicted_2%_train_obs_usgs.npy')
p_2_Mws_syn_test = np.load('./2%_syn_npy/predicted_2%_test_syn.npy')
p_2_Mws_syn_train = np.load('./2%_syn_npy/predicted_2%_train_syn.npy')

p_2_Mws_obs_train_above6 = p_2_Mws_obs_train[indices_obs_Mws_above6]
error_2_Mws_obs_train_above6 = error_2_Mws_obs_train[indices_obs_Mws_above6]
p_2_Mws_obs_test_above6 = p_2_Mws_obs_test[indices_obs_Mws_test_above6]
error_2_Mws_obs_test_above6 = error_2_Mws_obs_test[indices_obs_Mws_test_above6]

error_1_Mws_obs_test = np.load('./1%_obs_npy/predicted_1%_test_obs_usgs_errors.npy')
error_1_Mws_obs_train = np.load('./1%_obs_npy/predicted_1%_train_obs_usgs_errors.npy')
error_1_Mws_syn_test = np.load('./1%_syn_npy/predicted_1%_test_syn_errors.npy')
error_1_Mws_syn_train = np.load('./1%_syn_npy/predicted_1%_train_syn_errors.npy')

p_1_Mws_obs_test = np.load('./1%_obs_npy/predicted_1%_test_obs_usgs.npy')
p_1_Mws_obs_train = np.load('./1%_obs_npy/predicted_1%_train_obs_usgs.npy')
p_1_Mws_syn_test = np.load('./1%_syn_npy/predicted_1%_test_syn.npy')
p_1_Mws_syn_train = np.load('./1%_syn_npy/predicted_1%_train_syn.npy')

p_1_Mws_obs_train_above6 = p_1_Mws_obs_train[indices_obs_Mws_above6]
error_1_Mws_obs_train_above6 = error_1_Mws_obs_train[indices_obs_Mws_above6]
p_1_Mws_obs_test_above6 = p_1_Mws_obs_test[indices_obs_Mws_test_above6]
error_1_Mws_obs_test_above6 = error_1_Mws_obs_test[indices_obs_Mws_test_above6]

# 定义范围
lower_bound = -0.3
upper_bound = 0.3

num_count_20_obs_train = np.count_nonzero((error_20_Mws_obs_train >= lower_bound) & (error_20_Mws_obs_train <= upper_bound))
rate_20_obs_train = num_count_20_obs_train/len(error_20_Mws_obs_train)
print(rate_20_obs_train)

num_count_20_obs_train_above6 = np.count_nonzero((error_20_Mws_obs_train_above6 >= lower_bound) & (error_20_Mws_obs_train_above6 <= upper_bound))
rate_20_obs_train_above6 = num_count_20_obs_train_above6/len(error_20_Mws_obs_train_above6)
print(rate_20_obs_train_above6)

num_count_20_obs_test = np.count_nonzero((error_20_Mws_obs_test >= lower_bound) & (error_20_Mws_obs_test <= upper_bound))
rate_20_obs_test = num_count_20_obs_test/len(error_20_Mws_obs_test)
print(rate_20_obs_test)

num_count_20_obs_test_above6 = np.count_nonzero((error_20_Mws_obs_test_above6 >= lower_bound) & (error_20_Mws_obs_test_above6 <= upper_bound))
rate_20_obs_test_above6 = num_count_20_obs_test_above6/len(error_20_Mws_obs_test_above6)

num_count_20_syn_train = np.count_nonzero((error_20_Mws_syn_train >= lower_bound) & (error_20_Mws_syn_train <= upper_bound))
rate_20_syn_train = num_count_20_syn_train/len(error_20_Mws_syn_train)
print(rate_20_syn_train)

num_count_20_syn_test = np.count_nonzero((error_20_Mws_syn_test >= lower_bound) & (error_20_Mws_syn_test <= upper_bound))
rate_20_syn_test = num_count_20_syn_test/len(error_20_Mws_syn_test)
print(rate_20_syn_test)

num_count_10_obs_train = np.count_nonzero((error_10_Mws_obs_train >= lower_bound) & (error_10_Mws_obs_train <= upper_bound))
rate_10_obs_train = num_count_10_obs_train/len(error_10_Mws_obs_train)

num_count_10_obs_train_above6 = np.count_nonzero((error_10_Mws_obs_train_above6 >= lower_bound) & (error_10_Mws_obs_train_above6 <= upper_bound))
rate_10_obs_train_above6 = num_count_10_obs_train_above6/len(error_10_Mws_obs_train_above6)

num_count_10_obs_test = np.count_nonzero((error_10_Mws_obs_test >= lower_bound) & (error_10_Mws_obs_test <= upper_bound))
rate_10_obs_test = num_count_10_obs_test/len(error_10_Mws_obs_test)

num_count_10_obs_test_above6 = np.count_nonzero((error_10_Mws_obs_test_above6 >= lower_bound) & (error_10_Mws_obs_test_above6 <= upper_bound))
rate_10_obs_test_above6 = num_count_10_obs_test_above6/len(error_10_Mws_obs_test_above6)

num_count_10_syn_train = np.count_nonzero((error_10_Mws_syn_train >= lower_bound) & (error_10_Mws_syn_train <= upper_bound))
rate_10_syn_train = num_count_10_syn_train/len(error_10_Mws_syn_train)

num_count_10_syn_test = np.count_nonzero((error_10_Mws_syn_test >= lower_bound) & (error_10_Mws_syn_test <= upper_bound))
rate_10_syn_test = num_count_10_syn_test/len(error_10_Mws_syn_test)

num_count_4_obs_train = np.count_nonzero((error_4_Mws_obs_train >= lower_bound) & (error_4_Mws_obs_train <= upper_bound))
rate_4_obs_train = num_count_4_obs_train/len(error_4_Mws_obs_train)

num_count_4_obs_train_above6 = np.count_nonzero((error_4_Mws_obs_train_above6 >= lower_bound) & (error_4_Mws_obs_train_above6 <= upper_bound))
rate_4_obs_train_above6 = num_count_4_obs_train_above6/len(error_4_Mws_obs_train_above6)

num_count_4_obs_test = np.count_nonzero((error_4_Mws_obs_test >= lower_bound) & (error_4_Mws_obs_test <= upper_bound))
rate_4_obs_test = num_count_4_obs_test/len(error_4_Mws_obs_test)

num_count_4_obs_test_above6 = np.count_nonzero((error_4_Mws_obs_test_above6 >= lower_bound) & (error_4_Mws_obs_test_above6 <= upper_bound))
rate_4_obs_test_above6 = num_count_4_obs_test_above6/len(error_4_Mws_obs_test_above6)

num_count_4_syn_train = np.count_nonzero((error_4_Mws_syn_train >= lower_bound) & (error_4_Mws_syn_train <= upper_bound))
rate_4_syn_train = num_count_4_syn_train/len(error_4_Mws_syn_train)

num_count_4_syn_test = np.count_nonzero((error_4_Mws_syn_test >= lower_bound) & (error_4_Mws_syn_test <= upper_bound))
rate_4_syn_test = num_count_4_syn_test/len(error_4_Mws_syn_test)

num_count_2_obs_train = np.count_nonzero((error_2_Mws_obs_train >= lower_bound) & (error_2_Mws_obs_train <= upper_bound))
rate_2_obs_train = num_count_2_obs_train/len(error_2_Mws_obs_train)

num_count_2_obs_train_above6 = np.count_nonzero((error_2_Mws_obs_train_above6 >= lower_bound) & (error_2_Mws_obs_train_above6 <= upper_bound))
rate_2_obs_train_above6 = num_count_2_obs_train_above6/len(error_2_Mws_obs_train_above6)

num_count_2_obs_test = np.count_nonzero((error_2_Mws_obs_test >= lower_bound) & (error_2_Mws_obs_test <= upper_bound))
rate_2_obs_test = num_count_2_obs_test/len(error_2_Mws_obs_test)

num_count_2_obs_test_above6 = np.count_nonzero((error_2_Mws_obs_test_above6 >= lower_bound) & (error_2_Mws_obs_test_above6 <= upper_bound))
rate_2_obs_test_above6 = num_count_2_obs_test_above6/len(error_2_Mws_obs_test_above6)

num_count_2_syn_train = np.count_nonzero((error_2_Mws_syn_train >= lower_bound) & (error_2_Mws_syn_train <= upper_bound))
rate_2_syn_train = num_count_2_syn_train/len(error_2_Mws_syn_train)

num_count_2_syn_test = np.count_nonzero((error_2_Mws_syn_test >= lower_bound) & (error_2_Mws_syn_test <= upper_bound))
rate_2_syn_test = num_count_2_syn_test/len(error_2_Mws_syn_test)

num_count_1_obs_train = np.count_nonzero((error_1_Mws_obs_train >= lower_bound) & (error_1_Mws_obs_train <= upper_bound))
rate_1_obs_train = num_count_1_obs_train/len(error_1_Mws_obs_train)

num_count_1_obs_train_above6 = np.count_nonzero((error_1_Mws_obs_train_above6 >= lower_bound) & (error_1_Mws_obs_train_above6 <= upper_bound))
rate_1_obs_train_above6 = num_count_1_obs_train_above6/len(error_1_Mws_obs_train_above6)

num_count_1_obs_test = np.count_nonzero((error_1_Mws_obs_test >= lower_bound) & (error_1_Mws_obs_test <= upper_bound))
rate_1_obs_test = num_count_1_obs_test/len(error_1_Mws_obs_test)

num_count_1_obs_test_above6 = np.count_nonzero((error_1_Mws_obs_test_above6 >= lower_bound) & (error_1_Mws_obs_test_above6 <= upper_bound))
rate_1_obs_test_above6 = num_count_1_obs_test_above6/len(error_1_Mws_obs_test_above6)

num_count_1_syn_train = np.count_nonzero((error_1_Mws_syn_train >= lower_bound) & (error_1_Mws_syn_train <= upper_bound))
rate_1_syn_train = num_count_1_syn_train/len(error_1_Mws_syn_train)

num_count_1_syn_test = np.count_nonzero((error_1_Mws_syn_test >= lower_bound) & (error_1_Mws_syn_test <= upper_bound))
rate_1_syn_test = num_count_1_syn_test/len(error_1_Mws_syn_test)

t = [1, 2, 4, 10, 20]
acc_obs_test = [rate_1_obs_test, rate_2_obs_test, rate_4_obs_test, rate_10_obs_test, rate_20_obs_test]
acc_obs_train = [rate_1_obs_train, rate_2_obs_train, rate_4_obs_train, rate_10_obs_train, rate_20_obs_train]

acc_obs_test_above6 = [rate_1_obs_test_above6, rate_2_obs_test_above6, rate_4_obs_test_above6, rate_10_obs_test_above6, rate_20_obs_test_above6]
acc_obs_train_above6 = [rate_1_obs_train_above6, rate_2_obs_train_above6, rate_4_obs_train_above6, rate_10_obs_train_above6, rate_20_obs_train_above6]

acc_syn_test = [rate_1_syn_test, rate_2_syn_test, rate_4_syn_test, rate_10_syn_test, rate_20_syn_test]
acc_syn_train = [rate_1_syn_train, rate_2_syn_train, rate_4_syn_train, rate_10_syn_train, rate_20_syn_train]

# 自定义格式化函数
def percentage_formatter(x, pos):
    return '{:.0%}'.format(x)

# 创建格式化器
formatter = FuncFormatter(percentage_formatter)

# 创建图形并设置子图布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))



sizes_obs = 20 * Mws_obs
sizes_syn = 20 * Mws_syn

sizes_obs_above6 = 20 * Mws_obs_above6



colors_20_obs_train = error_20_Mws_obs_train
colors_20_obs_test = error_20_Mws_obs_test
colors_20_syn_train = error_20_Mws_syn_train
colors_20_syn_test = error_20_Mws_syn_test

colors_20_obs_train_above6 = error_20_Mws_obs_train_above6

# scatter_plot = axes[0, 0].scatter(Mws_obs, p_20_Mws_obs_train, c=colors_20_obs_train, s=sizes_obs, alpha=1.0, cmap='magma', edgecolor='black')
scatter_plot = axes[0, 0].scatter(Mws_obs_above6, p_20_Mws_obs_train_above6, c=colors_20_obs_train_above6, s=sizes_obs_above6, alpha=1.0, cmap='magma', edgecolor='black')
colorbar = fig.colorbar(scatter_plot, ax=axes[0, 0])
colorbar.set_label(r'Error values', fontsize=25, fontweight='bold')
colorbar.ax.tick_params(labelsize=25, width=3.5)
colorbar.outline.set_linewidth(3.5)
x = [4.5, 10]
y = [4.5, 10]
# 计算误差线的偏移量
# 计算对角线的斜率和截距
diagonal_slope = (y[1] - y[0]) / (x[1] - x[0])
diagonal_intercept = y[0] - diagonal_slope * x[0]

# 计算平行线的斜率和截距（偏离10%）
parallel_slope = diagonal_slope
parallel_intercept_1 = diagonal_intercept + 0.3
parallel_intercept_2 = diagonal_intercept - 0.3

# 根据斜率和截距绘制平行线
x_parallel = np.linspace(4.5, 10.0, 1000)
y_parallel_1 = parallel_slope * x_parallel + parallel_intercept_1
y_parallel_2 = parallel_slope * x_parallel + parallel_intercept_2
axes[0, 0].plot(x, y, linewidth=5)  # 绘制对角线
axes[0, 0].plot(x_parallel, y_parallel_1, linestyle='--', color='red', linewidth=5)  # 绘制平行线1
axes[0, 0].plot(x_parallel, y_parallel_2, linestyle='--', color='blue', linewidth=5)  # 绘制平行线2


# 设置标题和标签
axes[0, 0].set_title("Real STFs", fontsize=35, fontweight="bold")
axes[0, 0].set_xlabel(r"True ${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 0].set_ylabel(r"Predicted ${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 0].set_xlim([5.5, 9.5])  # 设置 x 轴范围
axes[0, 0].set_ylim([5.5, 9.5])  # 设置 y 轴范围
axes[0, 0].tick_params(axis='y', labelcolor='k', labelsize=25, width=3.5)
axes[0, 0].tick_params(axis='x', labelcolor='k', labelsize=25, width=3.5)
axes[0, 0].spines['right'].set_linewidth(3.5)
axes[0, 0].spines['bottom'].set_linewidth(3.5)
axes[0, 0].spines['top'].set_linewidth(3.5)
axes[0, 0].spines['left'].set_linewidth(3.5)

fmt = '{:.1f}%'
rate_20_obs = rate_20_obs_train * 100
rate_20_obs_above6 = rate_20_obs_train_above6 * 100
rate_20_syn = rate_20_syn_train * 100
axes[0, 0].text(6.5, 8.5, fmt.format(rate_20_obs_above6), color='red', fontsize=30, fontweight="bold")
###########################################################################################################
scatter_plot1 = axes[0, 1].scatter(Mws_syn, p_20_Mws_syn_train, c=colors_20_syn_train, s=sizes_syn, alpha=1.0, cmap='viridis', edgecolor='black')
colorbar1 = fig.colorbar(scatter_plot1, ax=axes[0, 1])
colorbar1.set_label(r'Error values', fontsize=25, fontweight='bold')
colorbar1.ax.tick_params(labelsize=25, width=3.5)
colorbar1.outline.set_linewidth(3.5)
x = [4.5, 10]
y = [4.5, 10]
# 计算误差线的偏移量
# 计算对角线的斜率和截距
diagonal_slope = (y[1] - y[0]) / (x[1] - x[0])
diagonal_intercept = y[0] - diagonal_slope * x[0]

# 计算平行线的斜率和截距（偏离10%）
parallel_slope = diagonal_slope
parallel_intercept_1 = diagonal_intercept + 0.3
parallel_intercept_2 = diagonal_intercept - 0.3

# 根据斜率和截距绘制平行线
x_parallel = np.linspace(4.5, 10.0, 1000)
y_parallel_1 = parallel_slope * x_parallel + parallel_intercept_1
y_parallel_2 = parallel_slope * x_parallel + parallel_intercept_2
axes[0, 1].plot(x, y, linewidth=5)  # 绘制对角线
axes[0, 1].plot(x_parallel, y_parallel_1, linestyle='--', color='red', linewidth=5)  # 绘制平行线1
axes[0, 1].plot(x_parallel, y_parallel_2, linestyle='--', color='blue', linewidth=5)  # 绘制平行线2


# 设置标题和标签
axes[0, 1].set_title("Synthetic STFs", fontsize=35, fontweight="bold")
axes[0, 1].set_xlabel(r"True ${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 1].set_ylabel(r"Predicted ${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 1].set_xlim([6.8, 9.0])  # 设置 x 轴范围
axes[0, 1].set_ylim([6.8, 9.0])  # 设置 y 轴范围
axes[0, 1].yaxis.set_major_locator(MultipleLocator(0.5))
axes[0, 1].tick_params(axis='y', labelcolor='k', labelsize=25, width=3.5)
axes[0, 1].tick_params(axis='x', labelcolor='k', labelsize=25, width=3.5)
axes[0, 1].spines['right'].set_linewidth(3.5)
axes[0, 1].spines['bottom'].set_linewidth(3.5)
axes[0, 1].spines['top'].set_linewidth(3.5)
axes[0, 1].spines['left'].set_linewidth(3.5)

axes[0, 1].text(7.5, 8.5, fmt.format(rate_20_syn), color='red', fontsize=30, fontweight="bold")
######################################################################################################
axes[1, 0].axhline(y=0.8, color='red', linestyle='-', linewidth=10, alpha=0.8)

axes[1, 0].plot(t, acc_obs_test_above6, 'o-', color='turquoise', markersize=15, linewidth=5, label='Test real STFs')
axes[1, 0].plot(t, acc_obs_train_above6, 'v-', color='darkslategray', markersize=15, linewidth=5, label='Train real STFs')
axes[1, 0].fill_between(t, acc_obs_train_above6, color='darkslategray', alpha=0.1)
axes[1, 0].fill_between(t, acc_obs_test_above6, color='turquoise', alpha=0.3)
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
axes[1, 0].set_xlabel('Rupture processes', fontweight='bold', fontsize=25)
axes[1, 0].set_ylabel('Accuracy of MgNet', fontweight='bold', fontsize=25)
axes[1, 0].set_ylim(0.4, 1.0)
axes[1, 0].set_xticks([1, 2, 4, 10, 20])
axes[1, 0].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[1, 0].grid(True, linestyle='dashed')  # 显示网格
axes[1, 0].set_facecolor('whitesmoke')  # 设置背景颜色为灰色
axes[1, 0].tick_params(axis='y', labelcolor='k', labelsize=25, width=3.5)
axes[1, 0].tick_params(axis='x', labelcolor='k', labelsize=25, width=3.5)
axes[1, 0].spines['right'].set_linewidth(3.5)
axes[1, 0].spines['bottom'].set_linewidth(3.5)
axes[1, 0].spines['top'].set_linewidth(3.5)
axes[1, 0].spines['left'].set_linewidth(3.5)
axes[1, 0].legend(fontsize=20)
######################################################################################################
axes[1, 1].axhline(y=0.98, color='red', linestyle='-', linewidth=10, alpha=0.8)
axes[1, 1].plot(t, acc_syn_test, '*-', color='gold', markersize=15, linewidth=5, label='Test syn STFs')
axes[1, 1].plot(t, acc_syn_train, 'X-', color='darkviolet', markersize=15, linewidth=5, label='Train syn STFs')
axes[1, 1].fill_between(t, acc_syn_train, color='darkviolet', alpha=0.1)
axes[1, 1].fill_between(t, acc_syn_test, color='gold', alpha=0.3)
axes[1, 1].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
axes[1, 1].set_xlabel('Rupture processes', fontweight='bold', fontsize=25)
axes[1, 1].set_ylabel('Accuracy of MgNet', fontweight='bold', fontsize=25)
axes[1, 1].set_ylim(0.97, 1.01)
axes[1, 1].set_xticks([1, 2, 4, 10, 20])
axes[1, 1].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[1, 1].set_yticks([0.98, 0.99, 1.0])
axes[1, 1].set_yticklabels(['98%', '99%', '100%'], rotation=0)# 展示图表
axes[1, 1].grid(True, linestyle='dashed')  # 显示网格
axes[1, 1].set_facecolor('whitesmoke')  # 设置背景颜色为灰色
axes[1, 1].tick_params(axis='y', labelcolor='k', labelsize=25, width=3.5)
axes[1, 1].tick_params(axis='x', labelcolor='k', labelsize=25, width=3.5)
axes[1, 1].spines['right'].set_linewidth(3.5)
axes[1, 1].spines['bottom'].set_linewidth(3.5)
axes[1, 1].spines['top'].set_linewidth(3.5)
axes[1, 1].spines['left'].set_linewidth(3.5)
axes[1, 1].legend(fontsize=20)

axes[0, 0].annotate('a', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[0, 1].annotate('b', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 0].annotate('c', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 1].annotate('d', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_3.pdf', format='pdf', dpi=600)
plt.show()

exit()
