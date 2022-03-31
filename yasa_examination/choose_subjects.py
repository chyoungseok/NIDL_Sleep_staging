import os

def choose_prepared_edf(ground_truth_path, edf_path):
    choose_sub = []
    
    # get the file list of each path
    ground_truth_file_names = os.listdir(ground_truth_path)
    edf_file_names = os.listdir(edf_path)
    
    # 1. Check subjects whether or not they really have edf files
    choose_edf = []
    for edf_subject in edf_file_names:
        temp_dir = os.path.join(edf_path, edf_subject)
        temp_dir_list = os.listdir(temp_dir)
        for name in temp_dir_list:
            if 'edf' in name:
                choose_edf.append(edf_subject)
    
    # 2. Select subjects listed on both hypnogram and edf
    #    - use choose_edf to be compared with the stage_names
    for hypno_subject in ground_truth_file_names:
        for edf_name in choose_edf:
            # get the first six characters from each name
            temp_hypno = hypno_subject[0:6]
            temp_edf = edf_name
            
            # compare the two six-length-characters
            how_many_correspond = 0
            for i in range(0,6): 
                if temp_hypno[i] == temp_edf[i]:
                    how_many_correspond += 1
            
            # choose the subject if corresponds at least five times
            if how_many_correspond >= 5:
                choose_sub.append(temp_edf)    
#     choose_sub = pd.Series(choose_sub)
    return choose_sub