# for error finding 

import os
import pandas as pd
import staging_yasa
from tqdm.notebook import tqdm

path_subjects_edf = 'D:\\USC\\Sleep dataset\\Samsung_data\\NEW_EDF_matched'
subjects_edf = os.listdir(path_subjects_edf)
subjects_edf.sort()
print(subjects_edf)

filt_flag = 0

for subject_edf in tqdm(subjects_edf, desc='Total Progress'):
    print('\n==================================================')
    staging_yasa.get_yasa_hypnogram(path_subjects_edf, subject_edf, filt_flag)