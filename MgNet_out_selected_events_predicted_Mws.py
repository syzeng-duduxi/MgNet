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

obs_STF_o = np.load('./select_events_npy/STF_select.npy')
obs_Mws = np.load('./select_events_npy/Mws_select.npy')
obs_Mws_Time = np.load('./select_events_npy/Mw_Time_select.npy')
obs_T_durations = np.load('./select_events_npy/T_durations_select.npy')

obs_STF = obs_STF_o/1e20

predict_1_Mws_select = np.load('./predict_1%_select_events_npy/predict_1%_Mws_select.npy')
predict_2_Mws_select = np.load('./predict_2%_select_events_npy/predict_2%_Mws_select.npy')
predict_4_Mws_select = np.load('./predict_4%_select_events_npy/predict_4%_Mws_select.npy')
predict_10_Mws_select = np.load('./predict_10%_select_events_npy/predict_10%_Mws_select.npy')
predict_20_Mws_select = np.load('./predict_20%_select_events_npy/predict_20%_Mws_select.npy')

Times_1_Mws_select = obs_T_durations * 0.01
Times_2_Mws_select = obs_T_durations * 0.02
Times_4_Mws_select = obs_T_durations * 0.04
Times_10_Mws_select = obs_T_durations * 0.1
Times_20_Mws_select = obs_T_durations * 0.2

event_t0 = [Times_1_Mws_select[0], Times_2_Mws_select[0], Times_4_Mws_select[0], Times_10_Mws_select[0], Times_20_Mws_select[0]]
event_t1 = [Times_1_Mws_select[1], Times_2_Mws_select[1], Times_4_Mws_select[1], Times_10_Mws_select[1], Times_20_Mws_select[1]]
event_t2 = [Times_1_Mws_select[2], Times_2_Mws_select[2], Times_4_Mws_select[2], Times_10_Mws_select[2], Times_20_Mws_select[2]]
event_t3 = [Times_1_Mws_select[3], Times_2_Mws_select[3], Times_4_Mws_select[3], Times_10_Mws_select[3], Times_20_Mws_select[3]]

event_Mw0 = [predict_1_Mws_select[0], predict_2_Mws_select[0], predict_4_Mws_select[0], predict_10_Mws_select[0], predict_20_Mws_select[0]]
event_Mw1 = [predict_1_Mws_select[1], predict_2_Mws_select[1], predict_4_Mws_select[1], predict_10_Mws_select[1], predict_20_Mws_select[1]]
event_Mw2 = [predict_1_Mws_select[2], predict_2_Mws_select[2], predict_4_Mws_select[2], predict_10_Mws_select[2], predict_20_Mws_select[2]]
event_Mw3 = [predict_1_Mws_select[3], predict_2_Mws_select[3], predict_4_Mws_select[3], predict_10_Mws_select[3], predict_20_Mws_select[3]]

npts = 1000
T_STF_sample = []
n_STFs = 4
for i in np.arange(n_STFs):
    obs_T_durations_i = obs_T_durations[i]
    T_STF_sample_i = np.linspace(0, obs_T_durations_i, num=1000)
    T_STF_sample.append(T_STF_sample_i)

T_STF_sample = np.array(T_STF_sample)

# 创建图形并设置子图布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
#############################1
axes[0, 0].plot(T_STF_sample[1,:], obs_STF[1,:], color='darkslategray', linewidth=3.5, label='Moment rate')  # 绘制对角线
axes[0, 0].fill_between(T_STF_sample[1,:], obs_STF[1,:], color='darkslategray', alpha=0.1)

axes[0, 0].tick_params(axis='y', labelcolor='k', labelsize=20, width=2.5)
axes[0, 0].tick_params(axis='x', labelcolor='k', labelsize=20, width=2.5)
axes[0, 0].set_xlabel("Time after onset (s)", fontsize=20, fontweight="bold")
axes[0, 0].set_ylabel('Moment rate ($10^{20}$ Nm/s)', fontweight='bold', fontsize=20)
axes[0, 0].set_title(r'2011 Near East Coast Of Honshu Japan ${M}_{w}$ 9.0', fontweight='bold', fontsize=20)
axes[0, 0].spines['right'].set_linewidth(2.5)
axes[0, 0].spines['bottom'].set_linewidth(2.5)
axes[0, 0].spines['top'].set_linewidth(2.5)
axes[0, 0].spines['left'].set_linewidth(2.5)
axes[0, 0].set_ylim(0, max(obs_STF[1,:]*1.1))

ax2 = axes[0, 0].twinx()
ax2.axhline(y=obs_Mws[1], color='lightskyblue', linestyle='-', linewidth=25, alpha=0.8)
ax2.axvline(x=Times_1_Mws_select[1], color='red', linestyle='dashed', linewidth=3)
ax2.axvline(x=Times_2_Mws_select[1], color='red', linestyle='dashed', linewidth=3)
ax2.axvline(x=Times_4_Mws_select[1], color='red', linestyle='dashed', linewidth=3)
ax2.axvline(x=Times_10_Mws_select[1], color='red', linestyle='dashed', linewidth=3)
ax2.axvline(x=Times_20_Mws_select[1], color='red', linestyle='dashed', linewidth=3)
ax2.plot(T_STF_sample[1,:], obs_Mws_Time[1,:], '-', color='dodgerblue', linewidth=3.5, label=r'${M}_{w}$ evolution')
ax2.plot(event_t1, event_Mw1, '^-', color='orangered', markersize=10, linewidth=3.5, label=r'Predicted ${M}_{w}$')
ax2.fill_between(event_t1, event_Mw1, color='orangered', alpha=0.1)
ax2.set_ylim(0, obs_Mws[1]*1.1)
ax2.set_xlim(0, obs_T_durations[1])
ax2.tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
ax2.tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
ax2.set_ylabel('$M_w$', fontweight='bold', fontsize=20)

lines1, labels1 = axes[0, 0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

lines = lines1 + lines2
labels = labels1 + labels2

axes[0, 0].legend(lines, labels, fontsize=15, loc='lower right')

####################################################################
axes[0, 1].plot(T_STF_sample[3,:], obs_STF[3,:], color='darkslategray', linewidth=3.5, label='Moment rate')  # 绘制对角线
axes[0, 1].fill_between(T_STF_sample[3,:], obs_STF[3,:], color='darkslategray', alpha=0.1)

axes[0, 1].tick_params(axis='y', labelcolor='k', labelsize=20, width=2.5)
axes[0, 1].tick_params(axis='x', labelcolor='k', labelsize=20, width=2.5)
axes[0, 1].set_xlabel("Time after onset (s)", fontsize=20, fontweight="bold")
axes[0, 1].set_ylabel('Moment rate ($10^{20}$ Nm/s)', fontweight='bold', fontsize=20)
axes[0, 1].set_title(r'2013 Santa Cruz Islands ${M}_{w}$ 7.9', fontweight='bold', fontsize=20)
axes[0, 1].spines['right'].set_linewidth(2.5)
axes[0, 1].spines['bottom'].set_linewidth(2.5)
axes[0, 1].spines['top'].set_linewidth(2.5)
axes[0, 1].spines['left'].set_linewidth(2.5)
axes[0, 1].set_ylim(0, max(obs_STF[3,:]*1.1))
ax3 = axes[0, 1].twinx()

ax3.axhline(y=obs_Mws[3], color='lightskyblue', linestyle='-', linewidth=25, alpha=0.8)
ax3.axvline(x=Times_1_Mws_select[3], color='red', linestyle='dashed', linewidth=3)
ax3.axvline(x=Times_2_Mws_select[3], color='red', linestyle='dashed', linewidth=3)
ax3.axvline(x=Times_4_Mws_select[3], color='red', linestyle='dashed', linewidth=3)
ax3.axvline(x=Times_10_Mws_select[3], color='red', linestyle='dashed', linewidth=3)
ax3.axvline(x=Times_20_Mws_select[3], color='red', linestyle='dashed', linewidth=3)
ax3.plot(T_STF_sample[3,:], obs_Mws_Time[3,:], '-', color='dodgerblue', linewidth=3.5, label=r'${M}_{w}$ evolution')
ax3.plot(event_t3, event_Mw3, '^-', color='orangered', markersize=10, linewidth=3.5, label=r'Predicted ${M}_{w}$')
ax3.fill_between(event_t3, event_Mw3, color='orangered', alpha=0.1)
ax3.set_ylim(0, obs_Mws[3]*1.1)
ax3.set_xlim(0, obs_T_durations[3])
ax3.tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
ax3.tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
ax3.set_ylabel('$M_w$', fontweight='bold', fontsize=20)
################################################################
axes[1, 0].plot(T_STF_sample[2,:], obs_STF[2,:], color='darkslategray', linewidth=3.5, label='Moment rate')  # 绘制对角线
axes[1, 0].fill_between(T_STF_sample[2,:], obs_STF[2,:], color='darkslategray', alpha=0.1)

axes[1, 0].tick_params(axis='y', labelcolor='k', labelsize=20, width=2.5)
axes[1, 0].tick_params(axis='x', labelcolor='k', labelsize=20, width=2.5)
axes[1, 0].set_xlabel("Time after onset (s)", fontsize=20, fontweight="bold")
axes[1, 0].set_ylabel('Moment rate ($10^{20}$ Nm/s)', fontweight='bold', fontsize=20)
axes[1, 0].set_title(r'2012 Off W Coast Of Northern Sumatra ${M}_{w}$ 8.7', fontweight='bold', fontsize=20)
axes[1, 0].spines['right'].set_linewidth(2.5)
axes[1, 0].spines['bottom'].set_linewidth(2.5)
axes[1, 0].spines['top'].set_linewidth(2.5)
axes[1, 0].spines['left'].set_linewidth(2.5)
axes[1, 0].set_ylim(0, max(obs_STF[2,:]*1.1))
ax4 = axes[1, 0].twinx()

ax4.axhline(y=obs_Mws[2], color='lightskyblue', linestyle='-', linewidth=25, alpha=0.8)
ax4.axvline(x=Times_1_Mws_select[2], color='red', linestyle='dashed', linewidth=3)
ax4.axvline(x=Times_2_Mws_select[2], color='red', linestyle='dashed', linewidth=3)
ax4.axvline(x=Times_4_Mws_select[2], color='red', linestyle='dashed', linewidth=3)
ax4.axvline(x=Times_10_Mws_select[2], color='red', linestyle='dashed', linewidth=3)
ax4.axvline(x=Times_20_Mws_select[2], color='red', linestyle='dashed', linewidth=3)
ax4.plot(T_STF_sample[2,:], obs_Mws_Time[2,:], '-', color='dodgerblue', linewidth=3.5, label=r'${M}_{w}$ evolution')
ax4.plot(event_t2, event_Mw2, '^-', color='orangered', markersize=10, linewidth=3.5, label=r'Predicted ${M}_{w}$')
ax4.fill_between(event_t2, event_Mw2, color='orangered', alpha=0.1)
ax4.set_ylim(0, obs_Mws[2]*1.1)
ax4.set_xlim(0, obs_T_durations[2])
ax4.tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
ax4.tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
ax4.set_ylabel('$M_w$', fontweight='bold', fontsize=20)
###################################################################
axes[1, 1].plot(T_STF_sample[0,:], obs_STF[0,:], color='darkslategray', linewidth=3.5, label='Moment rate')  # 绘制对角线
axes[1, 1].fill_between(T_STF_sample[0,:], obs_STF[0,:], color='darkslategray', alpha=0.1)

axes[1, 1].tick_params(axis='y', labelcolor='k', labelsize=20, width=2.5)
axes[1, 1].tick_params(axis='x', labelcolor='k', labelsize=20, width=2.5)
axes[1, 1].set_xlabel("Time after onset (s)", fontsize=20, fontweight="bold")
axes[1, 1].set_ylabel('Moment rate ($10^{20}$ Nm/s)', fontweight='bold', fontsize=20)
axes[1, 1].set_title(r'2010 Near Coast Of Central Chile ${M}_{w}$ 8.8', fontweight='bold', fontsize=20)
axes[1, 1].spines['right'].set_linewidth(2.5)
axes[1, 1].spines['bottom'].set_linewidth(2.5)
axes[1, 1].spines['top'].set_linewidth(2.5)
axes[1, 1].spines['left'].set_linewidth(2.5)
axes[1, 1].set_ylim(0, max(obs_STF[0,:]*1.1))
ax5 = axes[1, 1].twinx()

ax5.axhline(y=obs_Mws[0], color='lightskyblue', linestyle='-', linewidth=25, alpha=0.8)
ax5.axvline(x=Times_1_Mws_select[0], color='red', linestyle='dashed', linewidth=3)
ax5.axvline(x=Times_2_Mws_select[0], color='red', linestyle='dashed', linewidth=3)
ax5.axvline(x=Times_4_Mws_select[0], color='red', linestyle='dashed', linewidth=3)
ax5.axvline(x=Times_10_Mws_select[0], color='red', linestyle='dashed', linewidth=3)
ax5.axvline(x=Times_20_Mws_select[0], color='red', linestyle='dashed', linewidth=3)
ax5.plot(T_STF_sample[0,:], obs_Mws_Time[0,:], '-', color='dodgerblue', linewidth=3.5, label=r'${M}_{w}$ evolution')
ax5.plot(event_t0, event_Mw0, '^-', color='orangered', markersize=10, linewidth=3.5, label=r'Predicted ${M}_{w}$')
ax5.fill_between(event_t0, event_Mw0, color='orangered', alpha=0.1)
ax5.set_ylim(0, obs_Mws[0]*1.1)
ax5.set_xlim(0, obs_T_durations[0])
ax5.tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
ax5.tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
ax5.set_ylabel('$M_w$', fontweight='bold', fontsize=20)

axes[0, 0].annotate('b', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[0, 1].annotate('c', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 0].annotate('d', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 1].annotate('e', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_2.pdf', format='pdf', dpi=600)
plt.show()

exit()
