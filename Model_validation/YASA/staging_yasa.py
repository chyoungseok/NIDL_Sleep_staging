import os
from turtle import goto
from mne.io.edf.edf import _edf_str
import pandas as pd
import mne
import yasa
import datetime

def str_time_to_seconds(time_str):
    time = time_str.split(':')
    seconds = int(time[0])*3600 + int(time[1])*60 + int(time[2])
    
    return seconds

def seconds_to_str_time(seconds):
    time = datetime.timedelta(seconds=seconds)
    return time

def get_yasa_hypnogram(path_subjects_edf, subject_edf, meas_time_df):

    subject = subject_edf[0:5]

    # load edf data
    now_edf = os.path.join(path_subjects_edf, subject_edf, 'Traces.edf')
    print('-- {}'.format(now_edf))
    raw = mne.io.read_raw_edf(now_edf, preload=True, verbose=False)

    # Channel Select
    eogs = ['no_eog', 'SO', 'SO-0', 'SO-1', 'IO']
    eegs = ['C3', 'C4', 'C3-A2', 'C4-A1']
    pick_chs = eogs+eegs
    raw.pick_channels(pick_chs)
    chs = raw.ch_names
    print("-- Available Channels: {}".format(chs))

    # start, end time matching
    # -- get start, end times
    hypno_start = meas_time_df.loc[subject[0:5], 'hypno_start']
    hypno_end = meas_time_df.loc[subject[0:5], 'hypno_end']
    edf_start = meas_time_df.loc[subject[0:5], 'edf_start']
    edf_end = meas_time_df.loc[subject[0:5], 'edf_end']

    hypno_start_sec = str_time_to_seconds(hypno_start)
    hypno_end_sec = str_time_to_seconds(hypno_end)
    edf_start_sec = str_time_to_seconds(edf_start)

    new_edf_start_sec = (hypno_start_sec - 30) - edf_start_sec
    new_edf_end_sec = hypno_end_sec - edf_start_sec

    print(f"-- Edf time before matching: {edf_start}  {edf_end}")
    print("-- Edf time after mathcing: {}  {}".format(
        str(seconds_to_str_time(edf_start_sec+new_edf_start_sec)),
        str(seconds_to_str_time(edf_start_sec+new_edf_end_sec))
    ))
    print(f"-- ground_truth time: {hypno_start}  {hypno_end}")

    # -- matching the time using raw.crop method
    raw.crop(tmin= new_edf_start_sec, tmax=new_edf_end_sec)

    os.mkdir('D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted\\'+subject)
    for eog in eogs:
        if eog == 'no_eog':
            eog = 'no_eog'
        elif not(eog in chs):
            continue

        for eeg in eegs:
            if not(eeg in chs):
                continue

            if eog == 'no_eog':
                sls = yasa.SleepStaging(raw, eeg_name=eeg)
            else:
                sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog)
            
            hypno_pred = sls.predict()
            hypno_pred = yasa.hypno_str_to_int(hypno_pred)
            yasa.plot_hypnogram(hypno_pred)

            # save as csv
            hypno_df = pd.DataFrame(hypno_pred, columns=['stages'])
            path_save = os.path.join('D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted',
            subject, '_'+eog+'_'+eeg+'.csv')
            hypno_df.to_csv(path_or_buf=path_save, index=None)


