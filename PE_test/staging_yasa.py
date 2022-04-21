from importlib.resources import path
import os
from turtle import goto
from mne.io.edf.edf import _edf_str
import pandas as pd
import mne
import yasa
import datetime
import seaborn as sns

def str_time_to_seconds(time_str):
    time = time_str.split(':')
    seconds = int(time[0])*3600 + int(time[1])*60 + int(time[2])
    
    return seconds

def seconds_to_str_time(seconds):
    time = datetime.timedelta(seconds=seconds)
    return time

def get_yasa_hypnogram(path_subjects_edf, subject_edf, filt_flag, path_save):
    # filt_flag = 0
    subject = subject_edf[0:8]

    # load edf data
    now_edf = os.path.join(path_subjects_edf, subject_edf)
    print('-- {}'.format(now_edf))
    raw = mne.io.read_raw_edf(now_edf, preload=True, verbose=False)

    # Channel Select
    eogs = ['no_eog', 'SO', 'SO-0', 'SO-1', 'IO']
    # eegs = ['C3', 'C4', 'C3-A2', 'C4-A1']
 
    eegs = 'C4'
    chs = raw.ch_names
    print(f"Available channels: {chs}")

    if filt_flag == 1:
        # Downsampling
        pre_sf = raw.info['sfreq'] # down sampling 하기 전의 sampling frequency
        raw.resample(100)
        post_sf = raw.info['sfreq'] # down sampling 이후의 sampling frequency
        print(f"Down-sampling done ... {pre_sf}Hz --> {post_sf}Hz")

        # Filtering
        raw.filter(0.3, 45) # 0.3 Hz ~ 45 Hz 로 band pass filtering
        print("Filtering done; Bandpass Filter [0.3 45]...")    

    # channel cropping 과 time matching은 VD_edf_matched를 생성하면서 이미 했음
    # 여기서 별도로 해 줄 필요 없음
    # LE dataset 또한, NEW_EDF_matched를 생성했기 때문에 위와 동일하게 접근 가능
    if filt_flag == 1:
        # path_save_subject = 'D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted_with_filt\\'+subject
        path_save_subject = os.path.join(path_save, 'predicted_with_filt', subject )
        path_save_subject_hypno = os.path.join(path_save_subject, 'pred_hypno')
        path_save_subject_prob = os.path.join(path_save_subject, 'prob')
    else:
        # path_save_subject = 'D:\\USC\\code_mine\\Model_validation\\YASA\\hypnograms\\predicted_without_filt\\'+subject
        path_save_subject = os.path.join(path_save, 'predicted_without_filt', subject )
        path_save_subject_hypno = os.path.join(path_save_subject, 'pred_hypno')
        path_save_subject_prob = os.path.join(path_save_subject, 'prob')

    try:
        os.mkdir(path_save_subject)           
        os.mkdir(path_save_subject_hypno)
        os.mkdir(path_save_subject_prob)
    except FileExistsError:
        pass

    
    # only C4만 사용하도록 함
    sls = yasa.SleepStaging(raw, eeg_name=eegs)
    hypno_pred = sls.predict()
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    prob_pred = sls.predict_proba()
    hypno_df = pd.DataFrame(hypno_pred, columns=['stages'])
    path_hypno = os.path.join(path_save_subject_hypno, 'pred_hypno_'+eegs+'.csv')
    path_prob = os.path.join(path_save_subject_prob, eegs+'_prob.csv')

    hypno_df.to_csv(path_or_buf=path_hypno, index=None)
    prob_pred.to_csv(path_or_buf=path_prob)


    # for eog in eogs:
    #     if eog == 'no_eog':
    #         eog = 'no_eog'
    #     elif not(eog in chs):
    #         continue

    #     for eeg in eegs:
    #         if not(eeg in chs):
    #             continue

    #         if eog == 'no_eog':
    #             sls = yasa.SleepStaging(raw, eeg_name=eeg)
    #         else:
    #             sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog)
            
    #         hypno_pred = sls.predict()
    #         hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    #         prob_pred = sls.predict_proba()
    #         #yasa.plot_hypnogram(hypno_pred)
    #         #sns.heatmap(prob_pred.transpose())

    #         # save as csv
    #         hypno_df = pd.DataFrame(hypno_pred, columns=['stages'])
    #         path_hypno = os.path.join(path_save_subject_hypno, '_'+eog+'_'+eeg+'.csv')
    #         path_prob = os.path.join(path_save_subject_prob, '_'+eog+'_'+eeg+'_prob.csv')

    #         hypno_df.to_csv(path_or_buf=path_hypno, index=None)
    #         prob_pred.to_csv(path_or_buf=path_prob)
    print("-- save done")


