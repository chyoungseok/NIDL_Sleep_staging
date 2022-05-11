import os
from tabnanny import verbose
import pandas as pd
import numpy as np
import mne
import yasa
from tqdm import tqdm

# mne.raw.info에 대한 정보 초기 선언
# ch_names = ['A1', 'A2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2']
ch_names = ['C4', 'A1']
sfreq = {'LE':500, 'LGD':500, 'MRN':500, 'PE':200, 'ST':500}
ch_types = ['eeg']*len(ch_names)

class txt2np:
    def __init__(self, path_txt, txt_subjects, sub_ID):
        self.path_txt = path_txt
        self.txt_subjects = txt_subjects
        self.sub_ID = sub_ID.upper()
        self.QNAN_dic = {}

    def txt_filenames(self):     
        # input: sub_ID
        # sub_ID를 포함하고 있는 txt 파일의 filename을 pd.Series로 반환 
        # output: a pd.Series that contains filenames of txt file with sub_ID  

        txt_subjects = self.txt_subjects
        sub_ID = self.sub_ID

        self.temp_txt_subjects = txt_subjects[txt_subjects.str.contains(sub_ID)]
        # if len(self.temp_txt_subjects) != 0:
            # print("\n\n-- now subject ID : {}, number of txt files : {}".format(sub_ID, len(self.temp_txt_subjects)))
            # print(self.temp_txt_subjects.values)

    def read_txt(self):
        # input: txt 파일의 filename이 담겨 있는 numpy array
        # f_names_txt에 담겨 있는 파일들을 읽어서, 하나의 numpy array로 반환
        # output: np_all_eeg, a numpy array that contains eeg data of all txt files in the f_names_txt

        f_names_txt = self.temp_txt_subjects
        np_all_eeg = np.array([])
        # for now_eeg in tqdm(ch_names, desc='Read txt files and Extract eeg data ... '):
        for now_eeg in ch_names:
            temp_eeg_txt = f_names_txt[f_names_txt.str.contains(now_eeg)].values[0]
            # print(f"now eeg = {now_eeg}, {temp_eeg_txt}")
            f = open(os.path.join(self.path_txt, temp_eeg_txt), encoding='ISO-8859-1')#'cp949')#'ISO-8859-1')
            lines = f.readlines()
            f.close()
            lines = lines[5:]

            if len(lines) == 0:
                # 비어있는 txt인 경우 pass
                continue

            temp_eeg = []
            for line in lines:
                temp_value = line.strip().split()[-1]
                if temp_value == '-1.#QNAN':
                    self.QNAN_dic[self.sub_ID] = now_eeg
                    temp_value = 0
                temp_eeg.append(temp_value)
            
            # Convert valuse in string into values as numpy array of float
            temp_eeg = np.array(temp_eeg, dtype=float)
            
            # C4와 A1의 길이가 다른 경우, 짧은 쪽으로 맞춰준다
            # 측정 시작은 함께 했으니, 시작 지점은 동일하다고 가정
            # 즉, 두 채널의 길이가 다른 경우, 길이가 더 긴 채널의 끝 부분을 자른다.
            if len(np_all_eeg)>0 and (len(np_all_eeg) != len(temp_eeg)):
                if len(np_all_eeg)>len(temp_eeg):
                    np_all_eeg = np_all_eeg[:-abs(len(np_all_eeg)-len(temp_eeg))]
                else:
                    temp_eeg = temp_eeg[:-abs(len(np_all_eeg)-len(temp_eeg))]

            np_all_eeg = np.concatenate((np_all_eeg, temp_eeg))
        if len(lines) != 0:
            # txt 파일이 비어있는 경우가 아닐 때, np_all_eeg 할당
            np_all_eeg = np_all_eeg.reshape([int(len(np_all_eeg)/len(temp_eeg)) , len(temp_eeg)])
        return np_all_eeg
        

class np2raw:
    def __init__(self, sub_ID, SOL):
        self.sub_ID = sub_ID.upper()
        self.SOL = SOL

    def np2raw(self, np_all_eeg):
        # input: np_all_eeg
        # output: mne.raw instance
        for key in sfreq:
            if key in self.sub_ID:
                temp_sfreq = sfreq[key]
                break
        info = mne.create_info(ch_names, temp_sfreq, ch_types, verbose=None)
        info['subject_info'] = {'his_id':self.sub_ID}
        self.raw = mne.io.RawArray(np_all_eeg, info, verbose=False)
        self.temp_sfreq = temp_sfreq

    def re_ref(self):
        if len(self.raw.ch_names) < 6:
            self.raw_re_ref = mne.set_bipolar_reference(self.raw, anode=['C4'], cathode=['A1'], copy=True, verbose=False)
        else:
            self.raw_re_ref = mne.set_bipolar_reference(self.raw, anode=['F3', 'F4', 'C3', 'C4', 'O1', 'O2'], cathode=['A2', 'A1', 'A2', 'A1', 'A2', 'A1'], copy=True, verbose=False)

    def raw_cropping(self):
        # load TIB and SOL
        # SOL = self.df_SOL.loc[self.sub_ID, 'Sleep latency (min)'
        self.raw_re_ref_cropped = self.raw_re_ref.copy().crop(tmin=self.SOL*60, tmax=self.SOL*60+330*60)
        return self.raw_re_ref_cropped

        

class automatic_staging:
    def __init__(self) -> None:
        pass

    def get_hypnos_and_probs(self, raw_re_ref_cropped, filt_flag=1):
        # input: mne.raw instance that is pre-processed (re-referencing, cropping)
        # output
        #  - ensamble_hypno_dic : 각 채널에 대한 predicted hypnogram이 dictionary에 담겨 있음
        #  - self.ensamble_prob_dic: 각 채널에 대한 probability hypnogram이 dictionary에 담겨 있음 
        #  - self.dic_key : 위 두 dictionary에 접근 하기 위한 key

        raw = raw_re_ref_cropped

        # Channel Assignment
        eogs = []
        eegs = ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1']
        chs = raw.ch_names
        # print(f"Available channels: {chs}")

        # Filtering
        if filt_flag == 1:
            # Downsampling
            pre_sf = raw.info['sfreq'] # down sampling 하기 전의 sampling frequency
            raw.resample(100,verbose=False)
            post_sf = raw.info['sfreq'] # down sampling 이후의 sampling frequency
            # print(f"Down-sampling done ... {pre_sf}Hz --> {post_sf}Hz")

            # Filtering
            raw.filter(0.3, 45, verbose=False) # 0.3 Hz ~ 45 Hz 로 band pass filtering
            # print("Filtering done; Bandpass Filter [0.3 45]...")    
        
        self.raw = raw
        # Automatic staging of each channel
        self.ensamble_hypno_dic = {} # 각 채널별 hyonogram을 저장하는 dictionary
        self.ensamble_prob_dic = {} # 각 채널별 confidence map을 저장하는 dictionary
        self.dic_key = [] # 두 dictionary에 접근 하기 위한 key를 저장하는 list

        # eeg만 보유하고 있기 때문에, eeg에 대해서만 for문 작성
        for eeg in eegs:
            if not(eeg in raw.ch_names):
                continue
            sls = yasa.SleepStaging(raw, eeg_name=eeg)

            temp_hypno = yasa.hypno_str_to_int(sls.predict())
            temp_prob = sls.predict_proba()
            temp_key = eeg

            self.dic_key.append(temp_key)
            self.ensamble_hypno_dic[temp_key] = temp_hypno
            self.ensamble_prob_dic[temp_key] = temp_prob     
        
        # update self.sub_ID
        self.sub_ID = raw.info['subject_info']['his_id']

    def ensamble_stagig(self, path_save):
        # for demo
        # -- txt_to_prob.py
        # -- hypnograms
        # -- -- subject_1
        # -- -- subject_2
        # -- -- ...
        # -- -- subject_n
        # -- -- -- predicted_hypnogram.csv
        # -- -- -- probabilistic_hypnogram.csv

        # for real data
        # -- E:\\probabilistic_hypnogram
        # -- -- subject_1
        # -- -- subject_2
        # -- -- ...
        # -- -- subject_n
        # -- -- -- predicted_hypnogram.csv
        # -- -- -- probabilistic_hypnogram.csv
        # path_save = 'E:\\probabilistic_hypnogram'

        sub_ID = self.sub_ID

        # Create directory '/hypnograms'
        try:
            os.mkdir(path_save)
        except FileExistsError:
            pass

        # Create directory '/hypnograms/sub_ID'
        try:
            os.mkdir(os.path.join(path_save, sub_ID))
        except FileExistsError:
            pass

        
        final_hypno = []    
        final_prob = self.ensamble_prob_dic[self.dic_key[0]]*0 # 모든 채널의 평균을 담을 그릇 생성 (pd.DataFrame)

        # sum of probabilistic hypnograms
        for key in self.dic_key: 
            final_prob += self.ensamble_prob_dic[key] # 각 채널의 probability hypnogram을 모두 sum

        # Find final predicted hypnogram after ensamble
        # --> epoch-by-epoch processing
        staging_str_to_num = {'W': 0, 'R': 4, 'N1': 1, 'N2':2, 'N3':3}
        # for epoch in tqdm(range(len(self.ensamble_hypno_dic[self.dic_key[0]])), desc='Apply ensamble to all combinations ...'):
        for epoch in range(len(self.ensamble_hypno_dic[self.dic_key[0]])):
            temp_epoch = final_prob.iloc[epoch]
            final_hypno.append(staging_str_to_num[temp_epoch.idxmax()])
        df_final_hypno = pd.DataFrame(final_hypno, columns=['stages'])
        
        # average of the sum of probabilistic hypnogram
        df_final_prob = final_prob / len(self.dic_key)

        # update self
        self.df_final_prob = df_final_prob
        self.df_final_hypno = df_final_hypno

        # save both df_final_hypno and df_final_prob as csv files
        df_final_hypno.to_csv(os.path.join(path_save, sub_ID, 'predicted_hypnogram.csv'), index=None)
        df_final_prob.to_csv(os.path.join(path_save, sub_ID, 'probabilistic_hypnogram.csv'))
        

