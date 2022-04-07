import os
import pandas as pd
import mne
import yasa
import datetime
import EDFlib

def str_time_to_seconds(time_str):
    time = time_str.split(':')
    seconds = int(time[0])*3600 + int(time[1])*60 + int(time[2])
    
    return seconds

def seconds_to_str_time(seconds):
    time = datetime.timedelta(seconds=seconds)
    return time

def do_it(path_subjects_edf, subject_edf, meas_time_df):

    subject = subject_edf[0:5]

    # load edf data
    now_edf = os.path.join(path_subjects_edf, subject_edf, 'Traces.edf')
    print('-- {}'.format(now_edf))
    raw = mne.io.read_raw_edf(now_edf, preload=True, verbose=False)

    # Channel Select
    eogs = ['SO', 'IO']
    eegs = ['C3', 'C4']
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

    # export as edf
    path_save = 'D:\\USC\\Sleep dataset\\Samsung_data\\NEW_EDF_matched'
    save_as = os.path.join(path_save, subject+'.edf')
    mne.export.export_raw(save_as, raw, fmt='edf')



