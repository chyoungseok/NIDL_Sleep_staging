import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# input
#   1. "path_C4_txts": directory where single channel eeg datas are located
#   2. "C4_txt": filename of single eeg data
#   3. "df_meas": DataFrame about measurement times for all subjects
#   4. "df_nan_sum": DataFrame to be updated

# output
#   1. "df_nan_sum": pd.DataFrame that contains the information of NaN values which occured during hypnogram 

def update_df(path_C4_txts, C4_txt, df_meas, df_nan_sum):

    df_meas_indexes = df_meas.index
    if not C4_txt[0:5] in df_meas_indexes:
        print("there is no edf and scoring for {}".format(C4_txt[0:5]))
        df_nan_sum.loc[C4_txt[0:5], 'hypno_start'] = 'no edf and ground_truth'
        return df_nan_sum

    temp_df_meas = df_meas.loc[C4_txt[0:5],:]
    hypno_start = temp_df_meas['hypno_start']
    edf_start = temp_df_meas['edf_start']
    sample_num_to_crop = (23 - int(edf_start[0:2]))*3600*500

    temp_txt = os.path.join(path_C4_txts, C4_txt)
    with open(temp_txt, 'r') as f:
        lines = f.readlines() # Read txt file
        lines = lines[sample_num_to_crop:] # include only valid information which contains the signal values
    
    # print(f"{lines[0]}")
    # print(f"length: {len(lines[0])}")
    if len(lines[0].split()) < 2:
        print("No time information --> pass")
        df_nan_sum.loc[C4_txt[0:5], 'nan_start'] = 'no time infromation'
        return df_nan_sum

    times = []    
    data = []
    for line in tqdm(lines, desc='    -- Extract eeg values ... '):
        temp_time = line.strip().split()[0]
        temp_data = line.strip().split()[1]
        times.append(temp_time)
        data.append(temp_data)

    print("    -- Conversion to pd.DataFrame ...")
    times_df = pd.DataFrame(times)
    data_df = pd.DataFrame(data)

    print("    -- Extract QNAN part ...")
    nan_condition = data_df[0] == '-1.#QNAN'
    print(f"    -- {nan_condition.sum()} of QNAN were detected !")

    nan_df = data_df[nan_condition].index

    if len(nan_df) == 0:
            print("    -- No QNAN detected !")
            return df_nan_sum

    idx_transition_start = [0]
    idx_transition_end = []
    for iter in range(len(nan_df)-1):
        if (nan_df[iter+1] - nan_df[iter]) > 10:
            idx_transition_end.append(iter)
            idx_transition_start.append(iter+1)
    if len(nan_df) > 0:
        idx_transition_end.append(len(nan_df)-1)        
    print(f"    -- {len(idx_transition_start)} segments of QNAN detected !")

    
    nan_start_times = times_df[0][nan_df[idx_transition_start]].values
    nan_end_times = times_df[0][nan_df[idx_transition_end]].values
    nan_times = np.append(nan_start_times, nan_end_times)

    # plt.plot(nan_df)
    # plt.xticks(idx_transition_start+idx_transition_end, nan_times, rotation=30)

    hypno_start_seconds = time_to_seconds(hypno_start)
    valid_nan_iter = 1
    print(f"\n    -- hypno start        nan_start           nan_end")
    for idx in range(len(idx_transition_start)):
        temp_nan_start = times_df[0][nan_df[idx_transition_start[idx]]]
        temp_nan_end = times_df[0][nan_df[idx_transition_end[idx]]]
        temp_nan_start_seconds = time_to_seconds(temp_nan_start)
        temp_nan_end_seconds = time_to_seconds(temp_nan_end)
        
        if temp_nan_start_seconds < hypno_start_seconds < temp_nan_end_seconds:
            print(f"    -- {hypno_start}            {temp_nan_start}    {temp_nan_end}")
            index_name = C4_txt[0:5]+'_'+str(valid_nan_iter)
            temp_nan_duration = str(datetime.timedelta(seconds = (temp_nan_end_seconds - temp_nan_start_seconds)))
            df_nan_sum.loc[index_name] = [hypno_start, temp_nan_start, temp_nan_end, temp_nan_duration]
            valid_nan_iter += 1
        else:
            print("    -- -                  -                   -")
    return df_nan_sum

def time_to_seconds(time):
    time = time.split(':')
    seconds = int(time[0])*3600 + int(time[1])*60 + int(time[2])
    return seconds