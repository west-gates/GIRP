import csv
import os
import pandas as pd
import time
import random
import cPickle as pickle
from regression_tree_cart_general import *


#Some preprocessing of input data
dm = pd.read_csv('dm_txt.csv')
dm = dm[['pid_visit','lbl_visit']]
df = pd.read_csv('df_txt.csv')
df_grp = df.groupby(['pid_contrb','var']).sum().reset_index()
df = df_grp[['pid_contrb','ctb_contrb','var']]
df['val'] = 1

#id_p is list of all prediction instance IDs
id_p = list(set(df['pid_contrb']))

trs, min_tr = cvt(df, dm, id_p, max_depth = 500,  Nmin = 10)

trs[min_tr].display_tree(save = True, filename = 'selected_tree.jpg', view=False,height=3000, width=5000)