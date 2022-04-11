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
    now_edf = os.path.join(path_subjects_edf, subject_edf)
    print('-- {}'.format(now_edf))
    raw = mne.io.read_raw_edf(now_edf, preload=True, verbose=False)

    # Channel Select
    eogs = ['no_eog', 'SO', 'SO-0', 'SO-1', 'IO']
    eegs = ['C3', 'C4', 'C3-A2', 'C4-A1']
    chs = raw.ch_names

    # channel cropping 과 time matching은 VD_edf_matched를 생성하면서 이미 했음
    # 여기서 별도로 해 줄 필요 없음
    # LE dataset 또한, NEW_EDF_matched를 생성했기 때문에 위와 동일하게 접근 가능
    try:
        os.mkdir('D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted\\'+subject)  
    except FileExistsError:
        pass

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
            # yasa.plot_hypnogram(hypno_pred)

            # save as csv
            hypno_df = pd.DataFrame(hypno_pred, columns=['stages'])
            path_save = os.path.join('D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted',
            subject, '_'+eog+'_'+eeg+'.csv')
            hypno_df.to_csv(path_or_buf=path_save, index=None)
    print("---- save done \n")


