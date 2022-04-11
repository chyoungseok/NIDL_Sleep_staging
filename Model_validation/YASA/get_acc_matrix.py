import numpy as np
import pandas as pd

def get_acc_matrix(ground_truth, predicted):
    # yasa.plot_hypnogram은 숫자로 표현된 array를 input으로 받음 {"Wake": 0, "REM": 4, "N1": 1, "N2": 2, "N3": 3}
    # Initialization of matrix
    hypno_g = ground_truth
    hypno_p = predicted
    
    index_names = ['Wake', 'R', 'N1', 'N2', 'N3']
    empty_dic = {
        'Wake': np.zeros(5),
        'R': np.zeros(5),
        'N1': np.zeros(5),
        'N2': np.zeros(5),
        'N3': np.zeros(5)
    }
    matrix = pd.DataFrame(empty_dic, index=index_names) # empty state
    
    len_Wake = len(hypno_g[hypno_g==0]) + 1
    len_R = len(hypno_g[hypno_g==4]) + 1
    len_N1 = len(hypno_g[hypno_g==1]) + 1
    len_N2 = len(hypno_g[hypno_g==2]) + 1
    len_N3 = len(hypno_g[hypno_g==3]) + 1
    
    matrix.loc['Wake', 'Wake'] = round((hypno_p[hypno_g==0] == 0).sum() / len_Wake * 100, 1)
    matrix.loc['Wake', 'R'] = round((hypno_p[hypno_g==0] == 4).sum() / len_Wake * 100, 1)
    matrix.loc['Wake', 'N1'] = round((hypno_p[hypno_g==0] == 1).sum() / len_Wake * 100, 1)
    matrix.loc['Wake', 'N2'] = round((hypno_p[hypno_g==0] == 2).sum() / len_Wake * 100, 1)
    matrix.loc['Wake', 'N3'] = round((hypno_p[hypno_g==0] == 3).sum() / len_Wake * 100, 1)
    
    matrix.loc['R', 'Wake'] = round((hypno_p[hypno_g==4] == 0).sum() / len_R * 100, 1)
    matrix.loc['R', 'R'] = round((hypno_p[hypno_g==4] == 4).sum() / len_R * 100, 1)
    matrix.loc['R', 'N1'] = round((hypno_p[hypno_g==4] == 1).sum() / len_R * 100, 1)
    matrix.loc['R', 'N2'] = round((hypno_p[hypno_g==4] == 2).sum() / len_R * 100, 1)
    matrix.loc['R', 'N3'] = round((hypno_p[hypno_g==4] == 3).sum() / len_R * 100, 1)
    
    matrix.loc['N1', 'Wake'] = round((hypno_p[hypno_g==1] == 0).sum() / len_N1 * 100, 1)
    matrix.loc['N1', 'R'] = round((hypno_p[hypno_g==1] == 4).sum() / len_N1 * 100, 1)
    matrix.loc['N1', 'N1'] = round((hypno_p[hypno_g==1] == 1).sum() / len_N1 * 100, 1)
    matrix.loc['N1', 'N2'] = round((hypno_p[hypno_g==1] == 2).sum() / len_N1 * 100, 1)
    matrix.loc['N1', 'N3'] = round((hypno_p[hypno_g==1] == 3).sum() / len_N1 * 100, 1)
    
    matrix.loc['N2', 'Wake'] = round((hypno_p[hypno_g==2] == 0).sum() / len_N2 * 100, 1)
    matrix.loc['N2', 'R'] = round((hypno_p[hypno_g==2] == 4).sum() / len_N2 * 100, 1)
    matrix.loc['N2', 'N1'] = round((hypno_p[hypno_g==2] == 1).sum() / len_N2 * 100, 1)
    matrix.loc['N2', 'N2'] = round((hypno_p[hypno_g==2] == 2).sum() / len_N2 * 100, 1)
    matrix.loc['N2', 'N3'] = round((hypno_p[hypno_g==2] == 3).sum() / len_N2 * 100, 1)
    
    matrix.loc['N3', 'Wake'] = round((hypno_p[hypno_g==3] == 0).sum() / len_N3 * 100, 1)
    matrix.loc['N3', 'R'] = round((hypno_p[hypno_g==3] == 4).sum() / len_N3 * 100, 1)
    matrix.loc['N3', 'N1'] = round((hypno_p[hypno_g==3] == 1).sum() / len_N3 * 100, 1)
    matrix.loc['N3', 'N2'] = round((hypno_p[hypno_g==3] == 2).sum() / len_N3 * 100, 1)
    matrix.loc['N3', 'N3'] = round((hypno_p[hypno_g==3] == 3).sum() / len_N3 * 100, 1)   
    
    return matrix