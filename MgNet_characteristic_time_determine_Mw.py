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
Mws_obs_all = np.concatenate((Mws_obs, Mws_obs_test))

Times_obs_train = np.load('./scardec_usgs_obs/obs_train_T_durations.npy')
Times_obs_valid = np.load('./scardec_usgs_obs/obs_valid_T_durations.npy')
Times_obs = np.concatenate((Times_obs_train, Times_obs_valid))
Times_obs_test = np.load('./scardec_usgs_obs/obs_test_T_durations.npy')
Times_obs_all = np.concatenate((Times_obs, Times_obs_test))

Times_obs_all_new = Times_obs_all * 0.2
Times_obs_all_10 = Times_obs_all * 0.1
Times_obs_all_4 = Times_obs_all * 0.04
Times_obs_all_2 = Times_obs_all * 0.02
Times_obs_all_1 = Times_obs_all * 0.01

Mws_syn_train = np.load('./syn_Mw_npy_test/syn_train_Mws.npy')
Mws_syn_valid = np.load('./syn_Mw_npy_test/syn_valid_Mws.npy')
Mws_syn = np.concatenate((Mws_syn_train, Mws_syn_valid))
Mws_syn_test = np.load('./syn_Mw_npy_test/syn_test_Mws.npy')
Mws_syn_all = np.concatenate((Mws_syn, Mws_syn_test))
print(Mws_syn_all.shape)
#exit()

Times_syn_train = np.load('./syn_Mw_npy_test/syn_train_T_durations.npy')
Times_syn_valid = np.load('./syn_Mw_npy_test/syn_valid_T_durations.npy')
Times_syn = np.concatenate((Times_syn_train, Times_syn_valid))
Times_syn_test = np.load('./syn_Mw_npy_test/syn_test_T_durations.npy')
Times_syn_all = np.concatenate((Times_syn, Times_syn_test))

Times_syn_all_new = Times_syn_all * 0.2
Times_syn_all_10 = Times_syn_all * 0.1
Times_syn_all_4 = Times_syn_all * 0.04
Times_syn_all_2 = Times_syn_all * 0.02
Times_syn_all_1 = Times_syn_all * 0.01

mask1 = (Mws_obs_all >= 5.5) & (Mws_obs_all < 6.0)
Mws_obs_all1 = Mws_obs_all[mask1]
indices_obs_Mws1 = np.where(mask1)[0]

mask2 = (Mws_obs_all >= 6.0) & (Mws_obs_all < 7.0)
Mws_obs_all2 = Mws_obs_all[mask2]
indices_obs_Mws2 = np.where(mask2)[0]

mask3 = (Mws_obs_all >= 7.0) & (Mws_obs_all < 8.0)
Mws_obs_all3 = Mws_obs_all[mask3]
indices_obs_Mws3 = np.where(mask3)[0]

mask4 = (Mws_obs_all >= 8.0) & (Mws_obs_all < 9.5)
Mws_obs_all4 = Mws_obs_all[mask4]
indices_obs_Mws4 = np.where(mask4)[0]
#
mask1_syn = (Mws_syn_all >= 7.0) & (Mws_syn_all < 7.5)
Mws_syn_all1 = Mws_syn_all[mask1_syn]
indices_syn_Mws1 = np.where(mask1_syn)[0]

mask2_syn = (Mws_syn_all >= 7.5) & (Mws_syn_all < 8.0)
Mws_syn_all2 = Mws_syn_all[mask2_syn]
indices_syn_Mws2 = np.where(mask2_syn)[0]

mask3_syn = (Mws_syn_all >= 8.0) & (Mws_syn_all < 8.5)
Mws_syn_all3 = Mws_syn_all[mask3_syn]
indices_syn_Mws3 = np.where(mask3_syn)[0]

mask4_syn = (Mws_syn_all >= 8.5) & (Mws_syn_all < 9.0)
Mws_syn_all4 = Mws_syn_all[mask4_syn]
indices_syn_Mws4 = np.where(mask4_syn)[0]

error_20_Mws_obs_test = np.load('./20%_obs_npy/predicted_20%_test_obs_usgs_errors.npy')
error_20_Mws_obs_train = np.load('./20%_obs_npy/predicted_20%_train_obs_usgs_errors.npy')
error_20_Mws_obs_all = np.concatenate((error_20_Mws_obs_train, error_20_Mws_obs_test))
error_20_Mws_syn_test = np.load('./20%_syn_npy/predicted_20%_test_syn_errors.npy')
error_20_Mws_syn_train = np.load('./20%_syn_npy/predicted_20%_train_syn_errors.npy')
error_20_Mws_syn_all = np.concatenate((error_20_Mws_syn_train, error_20_Mws_syn_test))

error_10_Mws_obs_test = np.load('./10%_obs_npy/predicted_10%_test_obs_usgs_errors.npy')
error_10_Mws_obs_train = np.load('./10%_obs_npy/predicted_10%_train_obs_usgs_errors.npy')
error_10_Mws_obs_all = np.concatenate((error_10_Mws_obs_train, error_10_Mws_obs_test))
error_10_Mws_syn_test = np.load('./10%_syn_npy/predicted_10%_test_syn_errors.npy')
error_10_Mws_syn_train = np.load('./10%_syn_npy/predicted_10%_train_syn_errors.npy')
error_10_Mws_syn_all = np.concatenate((error_10_Mws_syn_train, error_10_Mws_syn_test))

error_4_Mws_obs_test = np.load('./4%_obs_npy/predicted_4%_test_obs_usgs_errors.npy')
error_4_Mws_obs_train = np.load('./4%_obs_npy/predicted_4%_train_obs_usgs_errors.npy')
error_4_Mws_obs_all = np.concatenate((error_4_Mws_obs_train, error_4_Mws_obs_test))
error_4_Mws_syn_test = np.load('./4%_syn_npy/predicted_4%_test_syn_errors.npy')
error_4_Mws_syn_train = np.load('./4%_syn_npy/predicted_4%_train_syn_errors.npy')
error_4_Mws_syn_all = np.concatenate((error_4_Mws_syn_train, error_4_Mws_syn_test))

error_2_Mws_obs_test = np.load('./2%_obs_npy/predicted_2%_test_obs_usgs_errors.npy')
error_2_Mws_obs_train = np.load('./2%_obs_npy/predicted_2%_train_obs_usgs_errors.npy')
error_2_Mws_obs_all = np.concatenate((error_2_Mws_obs_train, error_2_Mws_obs_test))
error_2_Mws_syn_test = np.load('./2%_syn_npy/predicted_2%_test_syn_errors.npy')
error_2_Mws_syn_train = np.load('./2%_syn_npy/predicted_2%_train_syn_errors.npy')
error_2_Mws_syn_all = np.concatenate((error_2_Mws_syn_train, error_2_Mws_syn_test))

error_1_Mws_obs_test = np.load('./1%_obs_npy/predicted_1%_test_obs_usgs_errors.npy')
error_1_Mws_obs_train = np.load('./1%_obs_npy/predicted_1%_train_obs_usgs_errors.npy')
error_1_Mws_obs_all = np.concatenate((error_1_Mws_obs_train, error_1_Mws_obs_test))
error_1_Mws_syn_test = np.load('./1%_syn_npy/predicted_1%_test_syn_errors.npy')
error_1_Mws_syn_train = np.load('./1%_syn_npy/predicted_1%_train_syn_errors.npy')
error_1_Mws_syn_all = np.concatenate((error_1_Mws_syn_train, error_1_Mws_syn_test))

error_20_Mws1_obs_all = error_20_Mws_obs_all[indices_obs_Mws1]
error_20_Mws2_obs_all = error_20_Mws_obs_all[indices_obs_Mws2]
error_20_Mws3_obs_all = error_20_Mws_obs_all[indices_obs_Mws3]
error_20_Mws4_obs_all = error_20_Mws_obs_all[indices_obs_Mws4]

error_20_Mws1_syn_all = error_20_Mws_syn_all[indices_syn_Mws1]
error_20_Mws2_syn_all = error_20_Mws_syn_all[indices_syn_Mws2]
error_20_Mws3_syn_all = error_20_Mws_syn_all[indices_syn_Mws3]
error_20_Mws4_syn_all = error_20_Mws_syn_all[indices_syn_Mws4]

error_10_Mws1_obs_all = error_10_Mws_obs_all[indices_obs_Mws1]
error_10_Mws2_obs_all = error_10_Mws_obs_all[indices_obs_Mws2]
error_10_Mws3_obs_all = error_10_Mws_obs_all[indices_obs_Mws3]
error_10_Mws4_obs_all = error_10_Mws_obs_all[indices_obs_Mws4]

error_10_Mws1_syn_all = error_10_Mws_syn_all[indices_syn_Mws1]
error_10_Mws2_syn_all = error_10_Mws_syn_all[indices_syn_Mws2]
error_10_Mws3_syn_all = error_10_Mws_syn_all[indices_syn_Mws3]
error_10_Mws4_syn_all = error_10_Mws_syn_all[indices_syn_Mws4]

error_4_Mws1_obs_all = error_4_Mws_obs_all[indices_obs_Mws1]
error_4_Mws2_obs_all = error_4_Mws_obs_all[indices_obs_Mws2]
error_4_Mws3_obs_all = error_4_Mws_obs_all[indices_obs_Mws3]
error_4_Mws4_obs_all = error_4_Mws_obs_all[indices_obs_Mws4]

error_4_Mws1_syn_all = error_4_Mws_syn_all[indices_syn_Mws1]
error_4_Mws2_syn_all = error_4_Mws_syn_all[indices_syn_Mws2]
error_4_Mws3_syn_all = error_4_Mws_syn_all[indices_syn_Mws3]
error_4_Mws4_syn_all = error_4_Mws_syn_all[indices_syn_Mws4]

error_2_Mws1_obs_all = error_2_Mws_obs_all[indices_obs_Mws1]
error_2_Mws2_obs_all = error_2_Mws_obs_all[indices_obs_Mws2]
error_2_Mws3_obs_all = error_2_Mws_obs_all[indices_obs_Mws3]
error_2_Mws4_obs_all = error_2_Mws_obs_all[indices_obs_Mws4]

error_2_Mws1_syn_all = error_2_Mws_syn_all[indices_syn_Mws1]
error_2_Mws2_syn_all = error_2_Mws_syn_all[indices_syn_Mws2]
error_2_Mws3_syn_all = error_2_Mws_syn_all[indices_syn_Mws3]
error_2_Mws4_syn_all = error_2_Mws_syn_all[indices_syn_Mws4]

error_1_Mws1_obs_all = error_1_Mws_obs_all[indices_obs_Mws1]
error_1_Mws2_obs_all = error_1_Mws_obs_all[indices_obs_Mws2]
error_1_Mws3_obs_all = error_1_Mws_obs_all[indices_obs_Mws3]
error_1_Mws4_obs_all = error_1_Mws_obs_all[indices_obs_Mws4]

error_1_Mws1_syn_all = error_1_Mws_syn_all[indices_syn_Mws1]
error_1_Mws2_syn_all = error_1_Mws_syn_all[indices_syn_Mws2]
error_1_Mws3_syn_all = error_1_Mws_syn_all[indices_syn_Mws3]
error_1_Mws4_syn_all = error_1_Mws_syn_all[indices_syn_Mws4]

Time_20_Mws1_obs_all = Times_obs_all_new[indices_obs_Mws1]
Time_20_Mws2_obs_all = Times_obs_all_new[indices_obs_Mws2]
Time_20_Mws3_obs_all = Times_obs_all_new[indices_obs_Mws3]
Time_20_Mws4_obs_all = Times_obs_all_new[indices_obs_Mws4]

Time_20_Mws1_syn_all = Times_syn_all_new[indices_syn_Mws1]
Time_20_Mws2_syn_all = Times_syn_all_new[indices_syn_Mws2]
Time_20_Mws3_syn_all = Times_syn_all_new[indices_syn_Mws3]
Time_20_Mws4_syn_all = Times_syn_all_new[indices_syn_Mws4]

Time_10_Mws1_obs_all = Times_obs_all_10[indices_obs_Mws1]
Time_10_Mws2_obs_all = Times_obs_all_10[indices_obs_Mws2]
Time_10_Mws3_obs_all = Times_obs_all_10[indices_obs_Mws3]
Time_10_Mws4_obs_all = Times_obs_all_10[indices_obs_Mws4]

Time_10_Mws1_syn_all = Times_syn_all_10[indices_syn_Mws1]
Time_10_Mws2_syn_all = Times_syn_all_10[indices_syn_Mws2]
Time_10_Mws3_syn_all = Times_syn_all_10[indices_syn_Mws3]
Time_10_Mws4_syn_all = Times_syn_all_10[indices_syn_Mws4]

Time_4_Mws1_obs_all = Times_obs_all_4[indices_obs_Mws1]
Time_4_Mws2_obs_all = Times_obs_all_4[indices_obs_Mws2]
Time_4_Mws3_obs_all = Times_obs_all_4[indices_obs_Mws3]
Time_4_Mws4_obs_all = Times_obs_all_4[indices_obs_Mws4]

Time_4_Mws1_syn_all = Times_syn_all_4[indices_syn_Mws1]
Time_4_Mws2_syn_all = Times_syn_all_4[indices_syn_Mws2]
Time_4_Mws3_syn_all = Times_syn_all_4[indices_syn_Mws3]
Time_4_Mws4_syn_all = Times_syn_all_4[indices_syn_Mws4]

Time_2_Mws1_obs_all = Times_obs_all_2[indices_obs_Mws1]
Time_2_Mws2_obs_all = Times_obs_all_2[indices_obs_Mws2]
Time_2_Mws3_obs_all = Times_obs_all_2[indices_obs_Mws3]
Time_2_Mws4_obs_all = Times_obs_all_2[indices_obs_Mws4]

Time_2_Mws1_syn_all = Times_syn_all_2[indices_syn_Mws1]
Time_2_Mws2_syn_all = Times_syn_all_2[indices_syn_Mws2]
Time_2_Mws3_syn_all = Times_syn_all_2[indices_syn_Mws3]
Time_2_Mws4_syn_all = Times_syn_all_2[indices_syn_Mws4]

Time_1_Mws1_obs_all = Times_obs_all_1[indices_obs_Mws1]
Time_1_Mws2_obs_all = Times_obs_all_1[indices_obs_Mws2]
Time_1_Mws3_obs_all = Times_obs_all_1[indices_obs_Mws3]
Time_1_Mws4_obs_all = Times_obs_all_1[indices_obs_Mws4]

Time_1_Mws1_syn_all = Times_syn_all_1[indices_syn_Mws1]
Time_1_Mws2_syn_all = Times_syn_all_1[indices_syn_Mws2]
Time_1_Mws3_syn_all = Times_syn_all_1[indices_syn_Mws3]
Time_1_Mws4_syn_all = Times_syn_all_1[indices_syn_Mws4]

min_Time_obs_Mws1 = min(Time_20_Mws1_obs_all)
max_Time_obs_Mws1 = max(Time_20_Mws1_obs_all)

min_Time_obs_Mws2 = min(Time_20_Mws2_obs_all)
max_Time_obs_Mws2 = max(Time_20_Mws2_obs_all)

min_Time_obs_Mws3 = min(Time_20_Mws3_obs_all)
max_Time_obs_Mws3 = max(Time_20_Mws3_obs_all)

min_Time_obs_Mws4 = min(Time_20_Mws4_obs_all)
max_Time_obs_Mws4 = max(Time_20_Mws4_obs_all)

value_obs_range1 = f"{min_Time_obs_Mws1:.1f}-{max_Time_obs_Mws1:.1f}s"
value_obs_range2 = f"{min_Time_obs_Mws2:.1f}-{max_Time_obs_Mws2:.1f}s"
value_obs_range3 = f"{min_Time_obs_Mws3:.1f}-{max_Time_obs_Mws3:.1f}s"
value_obs_range4 = f"{min_Time_obs_Mws4:.1f}-{max_Time_obs_Mws4:.1f}s"

min_Time_obs_Mws1_10 = min(Time_10_Mws1_obs_all)
max_Time_obs_Mws1_10 = max(Time_10_Mws1_obs_all)

min_Time_obs_Mws2_10 = min(Time_10_Mws2_obs_all)
max_Time_obs_Mws2_10 = max(Time_10_Mws2_obs_all)

min_Time_obs_Mws3_10 = min(Time_10_Mws3_obs_all)
max_Time_obs_Mws3_10 = max(Time_10_Mws3_obs_all)

min_Time_obs_Mws4_10 = min(Time_10_Mws4_obs_all)
max_Time_obs_Mws4_10 = max(Time_10_Mws4_obs_all)

value_obs_range1_10 = f"{min_Time_obs_Mws1_10:.1f}-{max_Time_obs_Mws1_10:.1f}s"
value_obs_range2_10 = f"{min_Time_obs_Mws2_10:.1f}-{max_Time_obs_Mws2_10:.1f}s"
value_obs_range3_10 = f"{min_Time_obs_Mws3_10:.1f}-{max_Time_obs_Mws3_10:.1f}s"
value_obs_range4_10 = f"{min_Time_obs_Mws4_10:.1f}-{max_Time_obs_Mws4_10:.1f}s"

min_Time_obs_Mws1_4 = min(Time_4_Mws1_obs_all)
max_Time_obs_Mws1_4 = max(Time_4_Mws1_obs_all)

min_Time_obs_Mws2_4 = min(Time_4_Mws2_obs_all)
max_Time_obs_Mws2_4 = max(Time_4_Mws2_obs_all)

min_Time_obs_Mws3_4 = min(Time_4_Mws3_obs_all)
max_Time_obs_Mws3_4 = max(Time_4_Mws3_obs_all)

min_Time_obs_Mws4_4 = min(Time_4_Mws4_obs_all)
max_Time_obs_Mws4_4 = max(Time_4_Mws4_obs_all)

value_obs_range1_4 = f"{min_Time_obs_Mws1_4:.1f}-{max_Time_obs_Mws1_4:.1f}s"
value_obs_range2_4 = f"{min_Time_obs_Mws2_4:.1f}-{max_Time_obs_Mws2_4:.1f}s"
value_obs_range3_4 = f"{min_Time_obs_Mws3_4:.1f}-{max_Time_obs_Mws3_4:.1f}s"
value_obs_range4_4 = f"{min_Time_obs_Mws4_4:.1f}-{max_Time_obs_Mws4_4:.1f}s"

min_Time_obs_Mws1_2 = min(Time_2_Mws1_obs_all)
max_Time_obs_Mws1_2 = max(Time_2_Mws1_obs_all)

min_Time_obs_Mws2_2 = min(Time_2_Mws2_obs_all)
max_Time_obs_Mws2_2 = max(Time_2_Mws2_obs_all)

min_Time_obs_Mws3_2 = min(Time_2_Mws3_obs_all)
max_Time_obs_Mws3_2 = max(Time_2_Mws3_obs_all)

min_Time_obs_Mws4_2 = min(Time_2_Mws4_obs_all)
max_Time_obs_Mws4_2 = max(Time_2_Mws4_obs_all)

value_obs_range1_2 = f"{min_Time_obs_Mws1_2:.1f}-{max_Time_obs_Mws1_2:.1f}s"
value_obs_range2_2 = f"{min_Time_obs_Mws2_2:.1f}-{max_Time_obs_Mws2_2:.1f}s"
value_obs_range3_2 = f"{min_Time_obs_Mws3_2:.1f}-{max_Time_obs_Mws3_2:.1f}s"
value_obs_range4_2 = f"{min_Time_obs_Mws4_2:.1f}-{max_Time_obs_Mws4_2:.1f}s"

min_Time_obs_Mws1_1 = min(Time_1_Mws1_obs_all)
max_Time_obs_Mws1_1 = max(Time_1_Mws1_obs_all)

min_Time_obs_Mws2_1 = min(Time_1_Mws2_obs_all)
max_Time_obs_Mws2_1 = max(Time_1_Mws2_obs_all)

min_Time_obs_Mws3_1 = min(Time_1_Mws3_obs_all)
max_Time_obs_Mws3_1 = max(Time_1_Mws3_obs_all)

min_Time_obs_Mws4_1 = min(Time_1_Mws4_obs_all)
max_Time_obs_Mws4_1 = max(Time_1_Mws4_obs_all)

value_obs_range1_1 = f"{min_Time_obs_Mws1_1:.1f}-{max_Time_obs_Mws1_1:.1f}s"
value_obs_range2_1 = f"{min_Time_obs_Mws2_1:.1f}-{max_Time_obs_Mws2_1:.1f}s"
value_obs_range3_1 = f"{min_Time_obs_Mws3_1:.1f}-{max_Time_obs_Mws3_1:.1f}s"
value_obs_range4_1 = f"{min_Time_obs_Mws4_1:.1f}-{max_Time_obs_Mws4_1:.1f}s"

min_Time_syn_Mws1 = min(Time_20_Mws1_syn_all)
max_Time_syn_Mws1 = max(Time_20_Mws1_syn_all)

min_Time_syn_Mws2 = min(Time_20_Mws2_syn_all)
max_Time_syn_Mws2 = max(Time_20_Mws2_syn_all)

min_Time_syn_Mws3 = min(Time_20_Mws3_syn_all)
max_Time_syn_Mws3 = max(Time_20_Mws3_syn_all)

min_Time_syn_Mws4 = min(Time_20_Mws4_syn_all)
max_Time_syn_Mws4 = max(Time_20_Mws4_syn_all)

value_syn_range1 = f"{min_Time_syn_Mws1:.1f}-{max_Time_syn_Mws1:.1f}s"
value_syn_range2 = f"{min_Time_syn_Mws2:.1f}-{max_Time_syn_Mws2:.1f}s"
value_syn_range3 = f"{min_Time_syn_Mws3:.1f}-{max_Time_syn_Mws3:.1f}s"
value_syn_range4 = f"{min_Time_syn_Mws4:.1f}-{max_Time_syn_Mws4:.1f}s"

min_Time_syn_Mws1_10 = min(Time_10_Mws1_syn_all)
max_Time_syn_Mws1_10 = max(Time_10_Mws1_syn_all)

min_Time_syn_Mws2_10 = min(Time_10_Mws2_syn_all)
max_Time_syn_Mws2_10 = max(Time_10_Mws2_syn_all)

min_Time_syn_Mws3_10 = min(Time_10_Mws3_syn_all)
max_Time_syn_Mws3_10 = max(Time_10_Mws3_syn_all)

min_Time_syn_Mws4_10 = min(Time_10_Mws4_syn_all)
max_Time_syn_Mws4_10 = max(Time_10_Mws4_syn_all)

value_syn_range1_10 = f"{min_Time_syn_Mws1_10:.1f}-{max_Time_syn_Mws1_10:.1f}s"
value_syn_range2_10 = f"{min_Time_syn_Mws2_10:.1f}-{max_Time_syn_Mws2_10:.1f}s"
value_syn_range3_10 = f"{min_Time_syn_Mws3_10:.1f}-{max_Time_syn_Mws3_10:.1f}s"
value_syn_range4_10 = f"{min_Time_syn_Mws4_10:.1f}-{max_Time_syn_Mws4_10:.1f}s"

min_Time_syn_Mws1_4 = min(Time_4_Mws1_syn_all)
max_Time_syn_Mws1_4 = max(Time_4_Mws1_syn_all)

min_Time_syn_Mws2_4 = min(Time_4_Mws2_syn_all)
max_Time_syn_Mws2_4 = max(Time_4_Mws2_syn_all)

min_Time_syn_Mws3_4 = min(Time_4_Mws3_syn_all)
max_Time_syn_Mws3_4 = max(Time_4_Mws3_syn_all)

min_Time_syn_Mws4_4 = min(Time_4_Mws4_syn_all)
max_Time_syn_Mws4_4 = max(Time_4_Mws4_syn_all)

value_syn_range1_4 = f"{min_Time_syn_Mws1_4:.1f}-{max_Time_syn_Mws1_4:.1f}s"
value_syn_range2_4 = f"{min_Time_syn_Mws2_4:.1f}-{max_Time_syn_Mws2_4:.1f}s"
value_syn_range3_4 = f"{min_Time_syn_Mws3_4:.1f}-{max_Time_syn_Mws3_4:.1f}s"
value_syn_range4_4 = f"{min_Time_syn_Mws4_4:.1f}-{max_Time_syn_Mws4_4:.1f}s"

min_Time_syn_Mws1_2 = min(Time_2_Mws1_syn_all)
max_Time_syn_Mws1_2 = max(Time_2_Mws1_syn_all)

min_Time_syn_Mws2_2 = min(Time_2_Mws2_syn_all)
max_Time_syn_Mws2_2 = max(Time_2_Mws2_syn_all)

min_Time_syn_Mws3_2 = min(Time_2_Mws3_syn_all)
max_Time_syn_Mws3_2 = max(Time_2_Mws3_syn_all)

min_Time_syn_Mws4_2 = min(Time_2_Mws4_syn_all)
max_Time_syn_Mws4_2 = max(Time_2_Mws4_syn_all)

value_syn_range1_2 = f"{min_Time_syn_Mws1_2:.1f}-{max_Time_syn_Mws1_2:.1f}s"
value_syn_range2_2 = f"{min_Time_syn_Mws2_2:.1f}-{max_Time_syn_Mws2_2:.1f}s"
value_syn_range3_2 = f"{min_Time_syn_Mws3_2:.1f}-{max_Time_syn_Mws3_2:.1f}s"
value_syn_range4_2 = f"{min_Time_syn_Mws4_2:.1f}-{max_Time_syn_Mws4_2:.1f}s"

min_Time_syn_Mws1_1 = min(Time_1_Mws1_syn_all)
max_Time_syn_Mws1_1 = max(Time_1_Mws1_syn_all)

min_Time_syn_Mws2_1 = min(Time_1_Mws2_syn_all)
max_Time_syn_Mws2_1 = max(Time_1_Mws2_syn_all)

min_Time_syn_Mws3_1 = min(Time_1_Mws3_syn_all)
max_Time_syn_Mws3_1 = max(Time_1_Mws3_syn_all)

min_Time_syn_Mws4_1 = min(Time_1_Mws4_syn_all)
max_Time_syn_Mws4_1 = max(Time_1_Mws4_syn_all)

value_syn_range1_1 = f"{min_Time_syn_Mws1_1:.1f}-{max_Time_syn_Mws1_1:.1f}s"
value_syn_range2_1 = f"{min_Time_syn_Mws2_1:.1f}-{max_Time_syn_Mws2_1:.1f}s"
value_syn_range3_1 = f"{min_Time_syn_Mws3_1:.1f}-{max_Time_syn_Mws3_1:.1f}s"
value_syn_range4_1 = f"{min_Time_syn_Mws4_1:.1f}-{max_Time_syn_Mws4_1:.1f}s"

num_count1 = np.count_nonzero((error_20_Mws1_obs_all >= -0.3) & (error_20_Mws1_obs_all <= 0.3))
num_count2 = np.count_nonzero((error_20_Mws2_obs_all >= -0.3) & (error_20_Mws2_obs_all <= 0.3))
num_count3 = np.count_nonzero((error_20_Mws3_obs_all >= -0.3) & (error_20_Mws3_obs_all <= 0.3))
num_count4 = np.count_nonzero((error_20_Mws4_obs_all >= -0.3) & (error_20_Mws4_obs_all <= 0.3))

num_count1_10 = np.count_nonzero((error_10_Mws1_obs_all >= -0.3) & (error_10_Mws1_obs_all <= 0.3))
num_count2_10 = np.count_nonzero((error_10_Mws2_obs_all >= -0.3) & (error_10_Mws2_obs_all <= 0.3))
num_count3_10 = np.count_nonzero((error_10_Mws3_obs_all >= -0.3) & (error_10_Mws3_obs_all <= 0.3))
num_count4_10 = np.count_nonzero((error_10_Mws4_obs_all >= -0.3) & (error_10_Mws4_obs_all <= 0.3))

num_count1_4 = np.count_nonzero((error_4_Mws1_obs_all >= -0.3) & (error_4_Mws1_obs_all <= 0.3))
num_count2_4 = np.count_nonzero((error_4_Mws2_obs_all >= -0.3) & (error_4_Mws2_obs_all <= 0.3))
num_count3_4 = np.count_nonzero((error_4_Mws3_obs_all >= -0.3) & (error_4_Mws3_obs_all <= 0.3))
num_count4_4 = np.count_nonzero((error_4_Mws4_obs_all >= -0.3) & (error_4_Mws4_obs_all <= 0.3))

num_count1_2 = np.count_nonzero((error_2_Mws1_obs_all >= -0.3) & (error_2_Mws1_obs_all <= 0.3))
num_count2_2 = np.count_nonzero((error_2_Mws2_obs_all >= -0.3) & (error_2_Mws2_obs_all <= 0.3))
num_count3_2 = np.count_nonzero((error_2_Mws3_obs_all >= -0.3) & (error_2_Mws3_obs_all <= 0.3))
num_count4_2 = np.count_nonzero((error_2_Mws4_obs_all >= -0.3) & (error_2_Mws4_obs_all <= 0.3))

num_count1_1 = np.count_nonzero((error_1_Mws1_obs_all >= -0.3) & (error_1_Mws1_obs_all <= 0.3))
num_count2_1 = np.count_nonzero((error_1_Mws2_obs_all >= -0.3) & (error_1_Mws2_obs_all <= 0.3))
num_count3_1 = np.count_nonzero((error_1_Mws3_obs_all >= -0.3) & (error_1_Mws3_obs_all <= 0.3))
num_count4_1 = np.count_nonzero((error_1_Mws4_obs_all >= -0.3) & (error_1_Mws4_obs_all <= 0.3))

num_count1_syn = np.count_nonzero((error_20_Mws1_syn_all >= -0.3) & (error_20_Mws1_syn_all <= 0.3))
num_count2_syn = np.count_nonzero((error_20_Mws2_syn_all >= -0.3) & (error_20_Mws2_syn_all <= 0.3))
num_count3_syn = np.count_nonzero((error_20_Mws3_syn_all >= -0.3) & (error_20_Mws3_syn_all <= 0.3))
num_count4_syn = np.count_nonzero((error_20_Mws4_syn_all >= -0.3) & (error_20_Mws4_syn_all <= 0.3))

num_count1_syn_10 = np.count_nonzero((error_10_Mws1_syn_all >= -0.3) & (error_10_Mws1_syn_all <= 0.3))
num_count2_syn_10 = np.count_nonzero((error_10_Mws2_syn_all >= -0.3) & (error_10_Mws2_syn_all <= 0.3))
num_count3_syn_10 = np.count_nonzero((error_10_Mws3_syn_all >= -0.3) & (error_10_Mws3_syn_all <= 0.3))
num_count4_syn_10 = np.count_nonzero((error_10_Mws4_syn_all >= -0.3) & (error_10_Mws4_syn_all <= 0.3))

num_count1_syn_4 = np.count_nonzero((error_4_Mws1_syn_all >= -0.3) & (error_4_Mws1_syn_all <= 0.3))
num_count2_syn_4 = np.count_nonzero((error_4_Mws2_syn_all >= -0.3) & (error_4_Mws2_syn_all <= 0.3))
num_count3_syn_4 = np.count_nonzero((error_4_Mws3_syn_all >= -0.3) & (error_4_Mws3_syn_all <= 0.3))
num_count4_syn_4 = np.count_nonzero((error_4_Mws4_syn_all >= -0.3) & (error_4_Mws4_syn_all <= 0.3))

num_count1_syn_2 = np.count_nonzero((error_2_Mws1_syn_all >= -0.3) & (error_2_Mws1_syn_all <= 0.3))
num_count2_syn_2 = np.count_nonzero((error_2_Mws2_syn_all >= -0.3) & (error_2_Mws2_syn_all <= 0.3))
num_count3_syn_2 = np.count_nonzero((error_2_Mws3_syn_all >= -0.3) & (error_2_Mws3_syn_all <= 0.3))
num_count4_syn_2 = np.count_nonzero((error_2_Mws4_syn_all >= -0.3) & (error_2_Mws4_syn_all <= 0.3))

num_count1_syn_1 = np.count_nonzero((error_1_Mws1_syn_all >= -0.3) & (error_1_Mws1_syn_all <= 0.3))
num_count2_syn_1 = np.count_nonzero((error_1_Mws2_syn_all >= -0.3) & (error_1_Mws2_syn_all <= 0.3))
num_count3_syn_1 = np.count_nonzero((error_1_Mws3_syn_all >= -0.3) & (error_1_Mws3_syn_all <= 0.3))
num_count4_syn_1 = np.count_nonzero((error_1_Mws4_syn_all >= -0.3) & (error_1_Mws4_syn_all <= 0.3))

rate1=num_count1/len(error_20_Mws1_obs_all)
rate2=num_count2/len(error_20_Mws2_obs_all)
rate3=num_count3/len(error_20_Mws3_obs_all)
rate4=num_count4/len(error_20_Mws4_obs_all)

rate1_10=num_count1_10/len(error_10_Mws1_obs_all)
rate2_10=num_count2_10/len(error_10_Mws2_obs_all)
rate3_10=num_count3_10/len(error_10_Mws3_obs_all)
rate4_10=num_count4_10/len(error_10_Mws4_obs_all)

rate1_4=num_count1_4/len(error_4_Mws1_obs_all)
rate2_4=num_count2_4/len(error_4_Mws2_obs_all)
rate3_4=num_count3_4/len(error_4_Mws3_obs_all)
rate4_4=num_count4_4/len(error_4_Mws4_obs_all)

rate1_2=num_count1_2/len(error_2_Mws1_obs_all)
rate2_2=num_count2_2/len(error_2_Mws2_obs_all)
rate3_2=num_count3_2/len(error_2_Mws3_obs_all)
rate4_2=num_count4_2/len(error_2_Mws4_obs_all)

rate1_1=num_count1_1/len(error_1_Mws1_obs_all)
rate2_1=num_count2_1/len(error_1_Mws2_obs_all)
rate3_1=num_count3_1/len(error_1_Mws3_obs_all)
rate4_1=num_count4_1/len(error_1_Mws4_obs_all)

rate1_syn=num_count1_syn/len(error_20_Mws1_syn_all)
rate2_syn=num_count2_syn/len(error_20_Mws2_syn_all)
rate3_syn=num_count3_syn/len(error_20_Mws3_syn_all)
rate4_syn=num_count4_syn/len(error_20_Mws4_syn_all)

rate1_syn_10=num_count1_syn_10/len(error_10_Mws1_syn_all)
rate2_syn_10=num_count2_syn_10/len(error_10_Mws2_syn_all)
rate3_syn_10=num_count3_syn_10/len(error_10_Mws3_syn_all)
rate4_syn_10=num_count4_syn_10/len(error_10_Mws4_syn_all)

rate1_syn_4=num_count1_syn_4/len(error_4_Mws1_syn_all)
rate2_syn_4=num_count2_syn_4/len(error_4_Mws2_syn_all)
rate3_syn_4=num_count3_syn_4/len(error_4_Mws3_syn_all)
rate4_syn_4=num_count4_syn_4/len(error_4_Mws4_syn_all)

rate1_syn_2=num_count1_syn_2/len(error_2_Mws1_syn_all)
rate2_syn_2=num_count2_syn_2/len(error_2_Mws2_syn_all)
rate3_syn_2=num_count3_syn_2/len(error_2_Mws3_syn_all)
rate4_syn_2=num_count4_syn_2/len(error_2_Mws4_syn_all)

rate1_syn_1=num_count1_syn_1/len(error_1_Mws1_syn_all)
rate2_syn_1=num_count2_syn_1/len(error_1_Mws2_syn_all)
rate3_syn_1=num_count3_syn_1/len(error_1_Mws3_syn_all)
rate4_syn_1=num_count4_syn_1/len(error_1_Mws4_syn_all)

t = [1, 2, 4, 10, 20]
acc_obs1 = [rate1_1, rate1_2, rate1_4, rate1_10, rate1]
acc_obs2 = [rate2_1, rate2_2, rate2_4, rate2_10, rate2]
acc_obs3 = [rate3_1, rate3_2, rate3_4, rate3_10, rate3]
acc_obs4 = [rate4_1, rate4_2, rate4_4, rate4_10, rate4]

acc_syn1 = [rate1_syn_1, rate1_syn_2, rate1_syn_4, rate1_syn_10, rate1_syn]
acc_syn2 = [rate2_syn_1, rate2_syn_2, rate2_syn_4, rate2_syn_10, rate2_syn]
acc_syn3 = [rate3_syn_1, rate3_syn_2, rate3_syn_4, rate3_syn_10, rate3_syn]
acc_syn4 = [rate4_syn_1, rate4_syn_2, rate4_syn_4, rate4_syn_10, rate4_syn]

# 自定义格式化函数
def percentage_formatter(x, pos):
    return '{:.0%}'.format(x)

# 创建格式化器
formatter = FuncFormatter(percentage_formatter)

percentage1 = f"{rate1 * 100:.1f}%"
percentage2 = f"{rate2 * 100:.1f}%"
percentage3 = f"{rate3 * 100:.1f}%"
percentage4 = f"{rate4 * 100:.1f}%"

percentage1_syn = f"{rate1_syn * 100:.1f}%"
percentage2_syn = f"{rate2_syn * 100:.1f}%"
percentage3_syn = f"{rate3_syn * 100:.1f}%"
percentage4_syn = f"{rate4_syn * 100:.1f}%"

# 创建图形并设置子图布局
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 22))
#############################1
cat = ('5.5~6.0', '6.0~7.0', '7.0~8.0', '8.0~9.5')
cat_above6 = ('6.0~7.0', '7.0~8.0', '8.0~9.5')
fcolors = ('deepskyblue', 'springgreen', 'gold', 'wheat')
# fcolors_above6 = ('deepskyblue', 'springgreen', 'wheat')
fcolors_above6 = ('purple', 'palevioletred', 'lightyellow')
# fcolors_p = ('steelblue', 'mediumseagreen', 'olive', 'orange')
fcolors_p_above6 = ('indigo', 'mediumvioletred', 'gold')
num_list = [len(error_20_Mws1_obs_all), len(error_20_Mws2_obs_all), len(error_20_Mws3_obs_all), len(error_20_Mws4_obs_all)]
num_list_above6 = [len(error_20_Mws2_obs_all), len(error_20_Mws3_obs_all), len(error_20_Mws4_obs_all)]
num_list_p = [num_count1, num_count2, num_count3, num_count4]
num_list_p_above6 = [num_count2, num_count3, num_count4]
percents = [percentage1, percentage2, percentage3, percentage4]
percents_above6 = [percentage2, percentage3, percentage4]
total = sum(num_list)
total_above6 = sum(num_list_above6)
percentage1_obs = f'{(len(error_20_Mws1_obs_all) / total * 100):.1f}%'
print(percentage1_obs)
percentage2_obs = f'{(len(error_20_Mws2_obs_all) / total * 100):.1f}%'
print(percentage2_obs)
percentage3_obs = f'{(len(error_20_Mws3_obs_all) / total * 100):.1f}%'
print(percentage3_obs)
percentage4_obs = f'{(len(error_20_Mws4_obs_all) / total * 100):.1f}%'
print(percentage4_obs)

percentage2_obs_above6 = f'{(len(error_20_Mws2_obs_all) / total_above6 * 100):.1f}%'
print(percentage2_obs_above6)
percentage3_obs_above6 = f'{(len(error_20_Mws3_obs_all) / total_above6 * 100):.1f}%'
print(percentage3_obs_above6)
percentage4_obs_above6 = f'{(len(error_20_Mws4_obs_all) / total_above6 * 100):.1f}%'
print(percentage4_obs_above6)

value_obs_ranges = [value_obs_range1, value_obs_range2, value_obs_range3, value_obs_range4]
value_obs_ranges_above6 = [value_obs_range2, value_obs_range3, value_obs_range4]
bars = axes[0, 0].bar(cat_above6, num_list_above6, width=0.5, color=fcolors_above6, alpha=0.99)
bars_p = axes[0, 0].bar(cat_above6, num_list_p_above6, width=0.5, color=fcolors_p_above6, alpha=0.99)
for i, bar in enumerate(bars_p):
    bar.set_label(f"$M_w$ {cat_above6[i]} acc: {percents_above6[i]}")  # 设置每个条形图的标签
axes[0, 0].legend(fontsize=15)
axes[0, 0].set_title(r"Accuracy of Real STFs on ${M}_{w}$ (20%)", fontsize=25, fontweight="bold")
axes[0, 0].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[0, 0].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[0, 0].set_xlabel(r"${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 0].set_ylabel('Number of events', fontweight='bold', fontsize=25)
axes[0, 0].spines['right'].set_linewidth(3.5)
axes[0, 0].spines['bottom'].set_linewidth(3.5)
axes[0, 0].spines['top'].set_linewidth(3.5)
axes[0, 0].spines['left'].set_linewidth(3.5)
#############################2
cat_syn = ('7.0~7.5', '7.5~8.0', '8.0~8.5', '8.5~9.0')
fcolors_syn = ('royalblue', 'mediumseagreen', 'lawngreen', 'khaki')
fcolors_p_syn = ('midnightblue', 'darkcyan', 'limegreen', 'yellow')
num_list_syn = [len(error_20_Mws1_syn_all), len(error_20_Mws2_syn_all), len(error_20_Mws3_syn_all), len(error_20_Mws4_syn_all)]
num_list_p_syn = [num_count1_syn, num_count2_syn, num_count3_syn, num_count4_syn]
percents_syn = [percentage1_syn, percentage2_syn, percentage3_syn, percentage4_syn]
value_obs_ranges_syn = [value_syn_range1, value_syn_range2, value_syn_range3, value_syn_range4]
bars_syn = axes[0, 1].bar(cat_syn, num_list_syn, width=0.5, color=fcolors_syn, alpha=0.99)
bars_p_syn = axes[0, 1].bar(cat_syn, num_list_p_syn, width=0.5, color=fcolors_p_syn, alpha=0.99)
for i_syn, bar_syn in enumerate(bars_p_syn):
    bar_syn.set_label(f"$M_w$ {cat_syn[i_syn]} acc: {percents_syn[i_syn]}")  # 设置每个条形图的标签
axes[0, 1].legend(fontsize=15)
total_syn = sum(num_list_syn)
percentage1_syn = f'{(len(error_20_Mws1_syn_all) / total_syn * 100):.1f}%'
print(percentage1_syn)
percentage2_syn = f'{(len(error_20_Mws2_syn_all) / total_syn * 100):.1f}%'
print(percentage2_syn)
percentage3_syn = f'{(len(error_20_Mws3_syn_all) / total_syn * 100):.1f}%'
print(percentage3_syn)
percentage4_syn = f'{(len(error_20_Mws4_syn_all) / total_syn * 100):.1f}%'
print(percentage4_syn)
axes[0, 1].set_title(r"Accuracy of Synthetic STFs on ${M}_{w}$ (20%)", fontsize=25, fontweight="bold")
axes[0, 1].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[0, 1].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[0, 1].set_ylim(0, 2000)
axes[0, 1].set_xlabel(r"${M}_{w}$", fontsize=25, fontweight="bold")
axes[0, 1].set_ylabel('Number of events', fontweight='bold', fontsize=25)
axes[0, 1].spines['right'].set_linewidth(3.5)
axes[0, 1].spines['bottom'].set_linewidth(3.5)
axes[0, 1].spines['top'].set_linewidth(3.5)
axes[0, 1].spines['left'].set_linewidth(3.5)
##############################3
axes[1, 0].axvline(x=1, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 0].axvline(x=2, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 0].axvline(x=4, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 0].axvline(x=10, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 0].axvline(x=20, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 0].axhline(y=0.8, color='red', linestyle='-', linewidth=10, alpha=0.8)
axes[1, 0].plot(t, acc_obs2, 'o-', color='indigo', markersize=15, linewidth=5, label=r'${M}_{w}$: 6.0~7.0')
axes[1, 0].plot(t, acc_obs3, 'v-', color='mediumvioletred', markersize=15, linewidth=5, label=r'${M}_{w}$: 7.0~8.0')
axes[1, 0].plot(t, acc_obs4, 'X-', color='gold', markersize=15, linewidth=5, label=r'${M}_{w}$: 8.0~9.5')
axes[1, 0].set_xticks([1, 2, 4, 10, 20])
axes[1, 0].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[1, 0].grid(True, linestyle='dashed')  # 显示网格
axes[1, 0].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[1, 0].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
axes[1, 0].set_xlabel('Rupture processes', fontsize=25, fontweight="bold")
axes[1, 0].set_ylabel('Accuracy of MgNet', fontweight='bold', fontsize=25)
axes[1, 0].spines['right'].set_linewidth(3.5)
axes[1, 0].spines['bottom'].set_linewidth(3.5)
axes[1, 0].spines['top'].set_linewidth(3.5)
axes[1, 0].spines['left'].set_linewidth(3.5)
axes[1, 0].legend(fontsize=15)
###############################4
axes[1, 1].axvline(x=1, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 1].axvline(x=2, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 1].axvline(x=4, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 1].axvline(x=10, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 1].axvline(x=20, color='darkgrey', linestyle='--', linewidth=3.5)
axes[1, 1].axhline(y=0.98, color='red', linestyle='-', linewidth=10, alpha=0.8)
axes[1, 1].plot(t, acc_syn1, 'o-', color='midnightblue', markersize=15, linewidth=5, label=r'${M}_{w}$: 7.0~7.5')
axes[1, 1].plot(t, acc_syn2, 'v-', color='darkcyan', markersize=15, linewidth=5, label=r'${M}_{w}$: 7.5~8.0')
axes[1, 1].plot(t, acc_syn3, '^-', color='limegreen', markersize=15, linewidth=5, label=r'${M}_{w}$: 8.0~8.5')
axes[1, 1].plot(t, acc_syn4, 'X-', color='yellow', markersize=15, linewidth=5, label=r'${M}_{w}$: 8.5~9.0')
axes[1, 1].set_ylim(0.97, 1.01)
axes[1, 1].set_xticks([1, 2, 4, 10, 20])
axes[1, 1].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[1, 1].set_yticks([0.98, 0.99, 1.0])
axes[1, 1].set_yticklabels(['98%', '99%', '100%'], rotation=0)# 展示图表
axes[1, 1].grid(True, linestyle='dashed')  # 显示网格
axes[1, 1].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[1, 1].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[1, 1].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
axes[1, 1].set_xlabel('Rupture processes', fontsize=25, fontweight="bold")
axes[1, 1].set_ylabel('Accuracy of MgNet', fontweight='bold', fontsize=25)
axes[1, 1].spines['right'].set_linewidth(3.5)
axes[1, 1].spines['bottom'].set_linewidth(3.5)
axes[1, 1].spines['top'].set_linewidth(3.5)
axes[1, 1].spines['left'].set_linewidth(3.5)
axes[1, 1].legend(fontsize=15,loc='lower right')
###############################################5
consume_obs = {'$M_w$ : 5.5~6.0': (max_Time_obs_Mws1_1-min_Time_obs_Mws1_1,max_Time_obs_Mws1_2-min_Time_obs_Mws1_2,max_Time_obs_Mws1_4-min_Time_obs_Mws1_4,max_Time_obs_Mws1_10-min_Time_obs_Mws1_10,max_Time_obs_Mws1-min_Time_obs_Mws1),
           '$M_w$ : 6.0~7.0': (max_Time_obs_Mws2_1-min_Time_obs_Mws2_1,max_Time_obs_Mws2_2-min_Time_obs_Mws2_2,max_Time_obs_Mws2_4-min_Time_obs_Mws2_4,max_Time_obs_Mws2_10-min_Time_obs_Mws2_10,max_Time_obs_Mws2-min_Time_obs_Mws2),
           '$M_w$ : 7.0~8.0': (max_Time_obs_Mws3_1-min_Time_obs_Mws3_1,max_Time_obs_Mws3_2-min_Time_obs_Mws3_2,max_Time_obs_Mws3_4-min_Time_obs_Mws3_4,max_Time_obs_Mws3_10-min_Time_obs_Mws3_10,max_Time_obs_Mws3-min_Time_obs_Mws3),
           '$M_w$ : 8.0~9.5': (max_Time_obs_Mws4_1-min_Time_obs_Mws4_1,max_Time_obs_Mws4_2-min_Time_obs_Mws4_2,max_Time_obs_Mws4_4-min_Time_obs_Mws1_4,max_Time_obs_Mws4_10-min_Time_obs_Mws4_10,max_Time_obs_Mws4-min_Time_obs_Mws4)}
fcolors_obs = {'$M_w$ : 5.5~6.0': 'steelblue',
           '$M_w$ : 6.0~7.0': 'mediumseagreen',
           '$M_w$ : 7.0~8.0': 'olive',
           '$M_w$ : 8.0~9.5': 'orange'}

consume_obs_above6 = {
           '$M_w$ : 6.0~7.0': (max_Time_obs_Mws2_1-min_Time_obs_Mws2_1,max_Time_obs_Mws2_2-min_Time_obs_Mws2_2,max_Time_obs_Mws2_4-min_Time_obs_Mws2_4,max_Time_obs_Mws2_10-min_Time_obs_Mws2_10,max_Time_obs_Mws2-min_Time_obs_Mws2),
           '$M_w$ : 7.0~8.0': (max_Time_obs_Mws3_1-min_Time_obs_Mws3_1,max_Time_obs_Mws3_2-min_Time_obs_Mws3_2,max_Time_obs_Mws3_4-min_Time_obs_Mws3_4,max_Time_obs_Mws3_10-min_Time_obs_Mws3_10,max_Time_obs_Mws3-min_Time_obs_Mws3),
           '$M_w$ : 8.0~9.5': (max_Time_obs_Mws4_1-min_Time_obs_Mws4_1,max_Time_obs_Mws4_2-min_Time_obs_Mws4_2,max_Time_obs_Mws4_4-min_Time_obs_Mws1_4,max_Time_obs_Mws4_10-min_Time_obs_Mws4_10,max_Time_obs_Mws4-min_Time_obs_Mws4)}
fcolors_obs_above6 = {
           '$M_w$ : 6.0~7.0': 'indigo',
           '$M_w$ : 7.0~8.0': 'mediumvioletred',
           '$M_w$ : 8.0~9.5': 'gold'}

axes[2, 0].axvline(x=1, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 0].axvline(x=2, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 0].axvline(x=4, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 0].axvline(x=10, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 0].axvline(x=20, color='darkgrey', linestyle='--', linewidth=3.5)

bottom = [0,0,0,0,0]
for name,num in consume_obs_above6.items():
    bar = axes[2, 0].bar(t, num, width=0.8, color=fcolors_obs_above6[name], label=name, bottom=bottom)
    bottom = [bottom[i]+num[i] for i in range(len(t))]

axes[2, 0].text(t[0]-0.9, 5+1, value_obs_range2_1, color='indigo', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[0]-0.9, 5+6, value_obs_range3_1, color='mediumvioletred', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[0]-0.9, 5+11, value_obs_range4_1, color='gold', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[1]-0.9, 35+1, value_obs_range2_2, color='indigo', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[1]-0.9, 35+6, value_obs_range3_2, color='mediumvioletred', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[1]-0.9, 35+11, value_obs_range4_2, color='gold', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[2]-0.9, 25+1, value_obs_range2_4, color='indigo', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[2]-0.9, 25+6, value_obs_range3_4, color='mediumvioletred', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[2]-0.9, 25+11, value_obs_range4_4, color='gold', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[3]-0.9, 45+1, value_obs_range2_10, color='indigo', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[3]-0.9, 45+6, value_obs_range3_10, color='mediumvioletred', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[3]-0.9, 45+11, value_obs_range4_10, color='gold', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[4]-0.9, 95+1, value_obs_range2, color='indigo', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[4]-0.9, 95+6, value_obs_range3, color='mediumvioletred', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].text(t[4]-0.9, 95+11, value_obs_range4, color='gold', fontsize=13, fontweight="bold", rotation=40)
axes[2, 0].set_ylim(0.0, 130)
axes[2, 0].set_xticks([1, 2, 4, 10, 20])
axes[2, 0].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[2, 0].grid(True, linestyle='dashed')  # 显示网格
axes[2, 0].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[2, 0].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[2, 0].set_xlabel('Rupture processes', fontsize=25, fontweight="bold")
axes[2, 0].set_ylabel('Time for predicted $M_w$ (s)', fontweight='bold', fontsize=25)
axes[2, 0].spines['right'].set_linewidth(3.5)
axes[2, 0].spines['bottom'].set_linewidth(3.5)
axes[2, 0].spines['top'].set_linewidth(3.5)
axes[2, 0].spines['left'].set_linewidth(3.5)
axes[2, 0].legend(fontsize=15,loc='upper left')
###############################################6
consume_syn = {'$M_w$ : 7.0~7.5': (max_Time_syn_Mws1_1-min_Time_syn_Mws1_1,max_Time_syn_Mws1_2-min_Time_syn_Mws1_2,max_Time_syn_Mws1_4-min_Time_syn_Mws1_4,max_Time_syn_Mws1_10-min_Time_syn_Mws1_10,max_Time_syn_Mws1-min_Time_syn_Mws1),
           '$M_w$ : 7.5~8.0': (max_Time_syn_Mws2_1-min_Time_syn_Mws2_1,max_Time_syn_Mws2_2-min_Time_syn_Mws2_2,max_Time_syn_Mws2_4-min_Time_syn_Mws2_4,max_Time_syn_Mws2_10-min_Time_syn_Mws2_10,max_Time_syn_Mws2-min_Time_syn_Mws2),
           '$M_w$ : 8.0~8.5': (max_Time_syn_Mws3_1-min_Time_syn_Mws3_1,max_Time_syn_Mws3_2-min_Time_syn_Mws3_2,max_Time_syn_Mws3_4-min_Time_syn_Mws3_4,max_Time_syn_Mws3_10-min_Time_syn_Mws3_10,max_Time_syn_Mws3-min_Time_syn_Mws3),
           '$M_w$ : 8.5~9.0': (max_Time_syn_Mws4_1-min_Time_syn_Mws4_1,max_Time_syn_Mws4_2-min_Time_syn_Mws4_2,max_Time_syn_Mws4_4-min_Time_syn_Mws1_4,max_Time_syn_Mws4_10-min_Time_syn_Mws4_10,max_Time_syn_Mws4-min_Time_syn_Mws4)}
fcolors_syn = {'$M_w$ : 7.0~7.5': 'midnightblue',
           '$M_w$ : 7.5~8.0': 'darkcyan',
           '$M_w$ : 8.0~8.5': 'limegreen',
           '$M_w$ : 8.5~9.0': 'yellow'}

axes[2, 1].axvline(x=1, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 1].axvline(x=2, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 1].axvline(x=4, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 1].axvline(x=10, color='darkgrey', linestyle='--', linewidth=3.5)
axes[2, 1].axvline(x=20, color='darkgrey', linestyle='--', linewidth=3.5)

bottom_syn = [0,0,0,0,0]
for name_syn,num_syn in consume_syn.items():
    bar_syn = axes[2, 1].bar(t, num_syn, width=0.8, color=fcolors_syn[name_syn], label=name_syn, bottom=bottom_syn)
    bottom_syn = [bottom_syn[i]+num_syn[i] for i in range(len(t))]

axes[2, 1].text(t[0]-0.9, 5+1, value_syn_range1_1, color='midnightblue', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[0]-0.9, 5+6, value_syn_range2_1, color='darkcyan', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[0]-0.9, 5+11, value_syn_range3_1, color='limegreen', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[0]-0.9, 5+16, value_syn_range4_1, color='yellow', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[1]-0.9, 35+1, value_syn_range1_2, color='midnightblue', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[1]-0.9, 35+6, value_syn_range2_2, color='darkcyan', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[1]-0.9, 35+11, value_syn_range3_2, color='limegreen', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[1]-0.9, 35+16, value_syn_range4_2, color='yellow', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[2]-0.9, 20+1, value_syn_range1_4, color='midnightblue', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[2]-0.9, 20+6, value_syn_range2_4, color='darkcyan', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[2]-0.9, 20+11, value_syn_range3_4, color='limegreen', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[2]-0.9, 20+16, value_syn_range4_4, color='yellow', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[3]-0.9, 35+1, value_syn_range1_10, color='midnightblue', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[3]-0.9, 35+6, value_syn_range2_10, color='darkcyan', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[3]-0.9, 35+11, value_syn_range3_10, color='limegreen', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[3]-0.9, 35+16, value_syn_range4_10, color='yellow', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[4]-1, 65+1, value_syn_range1, color='midnightblue', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[4]-1, 65+6, value_syn_range2, color='darkcyan', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[4]-1, 65+11, value_syn_range3, color='limegreen', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].text(t[4]-1, 65+16, value_syn_range4, color='yellow', fontsize=13, fontweight="bold", rotation=40)
axes[2, 1].set_ylim(0.0, 95)
axes[2, 1].set_xticks([1, 2, 4, 10, 20])
axes[2, 1].set_xticklabels(['1%', '2%', '4%', '10%', '20%'], rotation=45)# 展示图表
axes[2, 1].grid(True, linestyle='dashed')  # 显示网格
axes[2, 1].tick_params(axis='y', labelcolor='k', labelsize=20, width=3.5)
axes[2, 1].tick_params(axis='x', labelcolor='k', labelsize=20, width=3.5)
axes[2, 1].set_xlabel('Rupture processes', fontsize=25, fontweight="bold")
axes[2, 1].set_ylabel('Time for predicted $M_w$ (s)', fontweight='bold', fontsize=25)
axes[2, 1].spines['right'].set_linewidth(3.5)
axes[2, 1].spines['bottom'].set_linewidth(3.5)
axes[2, 1].spines['top'].set_linewidth(3.5)
axes[2, 1].spines['left'].set_linewidth(3.5)
axes[2, 1].legend(fontsize=15,loc='upper left')
#####################################################
axes[0, 0].annotate('a', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[0, 1].annotate('b', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 0].annotate('c', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[1, 1].annotate('d', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[2, 0].annotate('e', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')
axes[2, 1].annotate('f', xy=(-0.1, 1.08), xycoords='axes fraction', fontsize=40, fontweight='bold')

labels_pie = ['$M_w$: 6.0~7.0', '$M_w$: 7.0~8.0', '$M_w$: 8.0~9.5']
sizes_pie = [74.2, 23.2, 2.6]
colors_pie = ['purple', 'palevioletred', 'lightyellow']

labels_pie2 = ['$M_w$: 7.0~7.5', '$M_w$: 7.5~8.0', '$M_w$: 8.0~8.5', '$M_w$: 8.5~9.0']
sizes_pie2 = [33.3, 21.9, 22.7, 22.1]
colors_pie2 = ['royalblue', 'mediumseagreen', 'lawngreen', 'khaki']

ax1 = fig.add_axes([0.32, 0.75, 0.15, 0.15])  # 在画布上的自定义位置
ax1.pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 18, 'weight': 'bold', 'color': 'red'})

ax2 = fig.add_axes([0.63, 0.8, 0.15, 0.15])  # 在画布上的自定义位置
ax2.pie(sizes_pie2, labels=labels_pie2, colors=colors_pie2, autopct='%1.1f%%', startangle=120, textprops={'fontsize': 18, 'weight': 'bold', 'color': 'red'})

plt.tight_layout()
# plt.savefig('fig3_test_extended.png', dpi=600)
plt.savefig('fig_4.pdf', format='pdf', dpi=600)
plt.show()

exit()
