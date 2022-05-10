import os
import pandas as pd
import txt_to_prob as tp

# 각 class 별, subject ID read
print("Load subject's ID")
# print('-- Read PSG_list2.xlsx')
df_subject_ID = pd.read_excel('D:\\USC\\code_mine\\txt_to_prob\\PSG_list2.xlsx')

pd_healthy = df_subject_ID.iloc[:,0].dropna().str.slice(stop=-3)
pd_OSA = df_subject_ID.iloc[:,1].dropna().str.slice(stop=-3)
pd_INS = df_subject_ID.iloc[:,2].dropna().str.slice(stop=-3)
pd_COMISA = df_subject_ID.iloc[:,3].dropna().str.slice(stop=-3)

pd_total = pd.concat([pd_healthy, pd_OSA, pd_INS, pd_COMISA])
print("-- Number of subjects: Healthy({}), OSA({}), Insomnia({}), COMISA({})"
      .format(len(pd_healthy), len(pd_OSA), len(pd_INS), len(pd_COMISA)))
print("-- Total: {}".format(len(pd_total)))

path_txt = 'E:\\samsung_original\\ensamble_test' # txt 파일이 저장되어 있는 경로
txt_subjects = pd.Series(os.listdir(path_txt)) # 모든 txt 파일의 filename이 담겨 있는 pd.Series

QNAN = []
for sub_ID in pd_total:
    # Read txt files and extract eeg data for all channels
    txt2np = tp.txt2np(path_txt, txt_subjects, sub_ID)
    txt2np.txt_filenames()
    if len(txt2np.temp_txt_subjects) == 0:
          continue
    np_all_eeg = txt2np.read_txt()
    QNAN.append(txt2np.QNAN_dic)

    # Create mne.raw instance 
    # - data: np_all_eeg
    # - raw.info['subject_info']['his_id']: subject ID
    # - preprocessing: re_referencing + cropping
    # - raw.ch_names: ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1']
    np2raw = tp.np2raw(sub_ID)
    np2raw.np2raw(np_all_eeg)
    np2raw.re_ref()
    raw_re_ref_cropped = np2raw.raw_cropping()

    # Automatic Sleep Staging
    # - apply ensamble using all six eeg channels
    # - automatiaclly saved
    #   current directory
    #   -- hypnograms
    #   -- -- subject_1
    #   -- -- subject_2
    #   -- -- ...
    #   -- -- subject_n
    #   -- -- -- predicted_hypnogram.csv
    #   -- -- -- probabilistic_hypnogram.csv
    automatic_staging = tp.automatic_staging()
    automatic_staging.get_hypnos_and_probs(raw_re_ref_cropped)
    automatic_staging.ensamble_stagig()

