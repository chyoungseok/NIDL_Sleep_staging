import os
import numpy as np
from tqdm.notebook import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import mne
import yasa
import datetime

import choose_subjects

def by_eeg_eog(now_subject_num):
    # data path 선언
#     ground_truth_path = 'D:\\USC\\Sleep dataset\\Samsung_data\\GROUND_TRUTH_STAGING'
#     edf_path = 'D:\\USC\\Sleep dataset\\Samsung_data\\REGULAR_EDF'
    
    ground_truth_path = 'G:\\다른 컴퓨터\\내 노트북\\USC\\Sleep dataset\\Samsung_data\\GROUND_TRUTH_STAGING'
    edf_path = 'G:\\다른 컴퓨터\\내 노트북\\USC\\Sleep dataset\\Samsung_data\\REGULAR_EDF'

    # get prepared_subjects using 'choose_subjects' module
    prepared_subjects = choose_subjects.choose_prepared_edf(ground_truth_path, edf_path)
    prepared_subjects.sort()
    print(prepared_subjects)
    print("\n-- Number of prepared subjects: {}".format(len(prepared_subjects)))

    # select a subject we want to analyze
    # now_subject_num = 8

    now_edf = os.path.join(edf_path, prepared_subjects[now_subject_num]) + '\\Traces.edf'
    print("Now edf: {}".format(now_edf))
    raw = mne.io.read_raw_edf(now_edf, preload=True)

    # remove the EOG, EMG, and EKG channels
    # raw.drop_channels(['ROC-A1', 'LOC-A2', 'EMG1-EMG2', 'EKG-R-EKG-L'])
    # eogs = ['no_eog', 'SO', 'SO-0', 'SO-1', 'IO']
    eogs = ['no_eog', 'LOC-0', 'LOC-1', 'ROC-0', 'ROC-1']
    if '박규희' in now_edf:
        eegs = ['C3', 'C4']
    else:
        eegs = ['C3-A2', 'C4-A1']
    raw.pick_channels(eegs + eogs)
    chan = raw.ch_names
    print("-- Available Channels: {}".format(chan))

    # data cropping
    # ground_truth_hypnogram과 edf 데이터의 시간이 다르므로, 이것을 match 시켜 준다

    # -- 1. load event.txt and get start and end time of ground-truth-labeling

#     ground_events_path = 'D:\\USC\\Sleep dataset\\Samsung_data\\GROUND_TRUTH_STAGING'
    print("Read EDF of {}".format(prepared_subjects[now_subject_num]))

    ground_events_path = 'G:\\다른 컴퓨터\\내 노트북\\USC\\Sleep dataset\\Samsung_data\\GROUND_TRUTH_STAGING'
    events = os.listdir(ground_events_path)
    # print(events)
    choose_idx = []
    i = 0
    for event in events:
        if prepared_subjects[now_subject_num][0:5] in event:
            choose_idx.append(i)
        i += 1
    print("\n -- Now subject is: {}".format(events[choose_idx[0]]))

    f = open(os.path.join(ground_events_path, events[choose_idx[0]]), 'r')
    lines = f.readlines()
    print(lines[17].strip().split())
    print(lines[18].strip().split())
    print(lines[len(lines)-1].strip().split())
    f.close

    hypno_start = lines[18].strip().split()[1]
    hypno_start_sec = int(hypno_start[0:2])*3600 + int(hypno_start[3:5])*60 + int(hypno_start[6:8])
    hypno_end = lines[len(lines)-1].strip().split()[1]
    hypno_end_sec = int(hypno_end[0:2])*3600 + int(hypno_end[3:5])*60 + int(hypno_end[6:8])

    print("\nhypno_start: {} ({} seconds)".format(hypno_start, hypno_start_sec))
    print("hypno_end: {} ({} seconds)".format(hypno_end, hypno_end_sec))

    # -- 2. get edf_start 
    edf_start_hour = raw.info['meas_date'].hour
    edf_start_min = raw.info['meas_date'].minute
    edf_start_sec = raw.info['meas_date'].second

    edf_start = (lambda x: '0'+x if len(x) < 8 else x)(str(edf_start_hour) + ':' + str(edf_start_min) + ':' + str(edf_start_sec))
    edf_start_sec = edf_start_hour*3600+edf_start_min*60+edf_start_sec

    print("\nedf_start: {} ({} seconds)".format(edf_start, edf_start_sec))

    edf_duration = raw.times[-1:]
    edf_end_sec = int(edf_start_sec + edf_duration)
    edf_end = (lambda x: '0'+x if len(x) < 8 else x)(str(datetime.timedelta(seconds=int(np.floor(edf_end_sec)))))
    print("edf_end: {} ({} seconds)".format(edf_end, edf_end_sec))

    if (hypno_start_sec < edf_start_sec) | (hypno_end_sec > edf_end_sec):
        print("Start or End time mismatch between ground-truth-hypno and edf file")
    else:
        # -- 3. cut the edf data to match with the hypnogram
        tmin = hypno_start_sec - edf_start_sec -30
        tmax = hypno_end_sec - edf_start_sec
        raw.crop(tmin=tmin, tmax=tmax)

        # Downsampling and filtering
        print("-- Original sampling rate: {}".format(raw.info['sfreq']))
        raw.resample(100)
        print("-- Sampling rate after downsampled: {}\n".format(raw.info['sfreq']))

        # 0.3-45 Hz bandpass-filter
        raw.filter(0.3, 45)

        # get single EEG data
        data = raw.get_data() * 1e6
        print("Shape of single EEG data: {}".format(data.shape))

        combination = []
        accuracy_record = []
    
        for eog in eogs:
            # print("now EOG: {}".format(eog))
            # if any(eog in s for s in chan):
            #     print("{} is available !".format(eog))
            if (eog == 'no_eog') | any(eog == s for s in chan):
                for eeg in eegs:
                    print("\n========================================================================")
                    print("========================================================================")
                    print("EOG: {}, EEG: {}".format(eog, eeg))
                    print(type(eeg), type(eog))
                    if eog == 'no_eog':
                        sls = yasa.SleepStaging(raw, eeg_name=eeg)
                    else:
                        sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name = eog)
                    hypno_pred = sls.predict()  # Predict the sleep stages
                    hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
                    # yasa.plot_hypnogram(hypno_pred);  # Plot

                    # convert hypnogram into pd.DataFrame and Save as csv
                    df_hypno = pd.DataFrame(hypno_pred, columns=['stages'])
                    # path_save = os.path.join('D:\\USC\\code_mine\\yasa_examination\\predicted_hypnogram', prepared_subjects[now_subject_num][0:5]) + '.csv'
                    path_save = os.path.join('G:\\다른 컴퓨터\\내 노트북\\USC\\code_mine\\yasa_examination\\predicted_hypnogram', prepared_subjects[now_subject_num][0:5]) + '.csv'
                    print("Save as: {}".format(path_save))
                    df_hypno.to_csv(path_or_buf=path_save, index=None)

                    # path_InNum_Hypnos = 'D:\\USC\\test_data\\Prepared_InNum_Hypnos'
                    path_InNum_Hypnos = 'G:\\다른 컴퓨터\\내 노트북\\USC\\test_data\\Prepared_InNum_Hypnos'
                    InNum_Hypnos = os.listdir(path_InNum_Hypnos)
                    InNum_Hypnos.sort()
                    print("-- Total prepared hypnograms: {}".format(InNum_Hypnos))
                    print("\n-- Now hypnogram: {}".format(InNum_Hypnos[now_subject_num]))

                    ground_truth_hypno = pd.read_csv(os.path.join(path_InNum_Hypnos, InNum_Hypnos[now_subject_num]), squeeze=True)
                    # ground_truth_hypno

                    yasa.plot_hypnogram(ground_truth_hypno)
                    plt.title(eog+'_'+eeg+'_'+'ground_truth',  fontsize=30)
                    yasa.plot_hypnogram(hypno_pred)
                    plt.title(eog+'_'+eeg+'_'+'predicted', fontsize=30)

                    print("Length of ground_truth: {}".format(len(ground_truth_hypno)))
                    print("Length of predicted: {}".format(len(hypno_pred)))
                    # path = 'D:\\USC\\code_mine\\yasa_examination\\predicted_hypnogram\\' + prepared_subjects[now_subject_num][0:5] + '.txt'
                    path = 'G:\\다른 컴퓨터\\내 노트북\\USC\\code_mine\\yasa_examination\\predicted_hypnogram\\' + prepared_subjects[now_subject_num][0:5] + '.txt'
                    f = open(path, 'wt')
                    f.writelines(['ground_truth' + '  ' + 'predicted\n'])
                    f.writelines([str(len(ground_truth_hypno)) + '  ' + str(len(hypno_pred))])
                    f.close()

                    from sklearn.metrics import accuracy_score
                    # print(f"The accuracy is {100 * accuracy_score(ground_truth_hypno, hypno_pred[0:643]):.3f}%")
                    print(f"The accuracy is {100 * accuracy_score(ground_truth_hypno, hypno_pred):.3f}%")

                    combination.append(eog+'_'+eeg)
                    accuracy_record.append(100 * accuracy_score(ground_truth_hypno, hypno_pred))
        df_accuracy = pd.DataFrame(accuracy_record, index=combination, columns=['Accuracy [%}'])
        return df_accuracy
