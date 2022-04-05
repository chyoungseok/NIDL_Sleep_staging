import os
from turtle import st 
import pandas as pd

def str2num(stages_str):
    # input: single hypnogram
    stages_int = []
    # dictionary to convert string to num
    stage_dic = {"Wake": 0, "REM": 4, "N1": 1, "N2": 2, "N3": 3}
    for stage in stages_str:
        stages_int.append(stage_dic[stage])
    return stages_int

def ReadTxtFiles(subject, path_subjects):
    # subject 한 명의 hypnogram을 읽어옴
    stages = []
    with open(os.path.join(path_subjects, subject+'\\Traces_hypnogram.txt'), 'r') as f:
        lines = f.readlines() # Read txt file
        lines = lines[2:] # include only valid information which contains the stages

        # From long strings, pick only 'stage' up
        for line in lines:
            temp_line = line.strip().split() # get the stages as string  
            stages.append(temp_line[0])
    return stages

def get_hypnos(subject, path_subjects):
    stages_str = ReadTxtFiles(subject, path_subjects)
    stages_int = str2num(stages_str)
    stages_int = pd.DataFrame(stages_int)
    return stages_int