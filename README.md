## Global Model Interpretation via Recursive Partitionin (GIRP)
This paper implements the menthod described in paper [Global Model Interpretation via Recursive Partitionin](https://arxiv.org/abs/1802.04253). The implementation is based on the code in this [repo](https://github.com/chandarb/Python-Regression-Tree-Forest). Please cite both works properly if you are using them in your work.

### Prepare data
You should prepare the input data as the example `dm_txt.csv` and `df_txt.csv` in this repo. 
`dm_txt.csv` contains the classification labels. Column `pid_visit` is the ID of each prediction instance. Column `lbl_visit` is the label of each prediction instance.
`df_txt.csv` contains the local contributions of input variables for each prediction instance. Column `ctb_contrb` is the local contributions. Column `pid_contrb` is the ID of each prediction instance. Column `var` is the names of input variables.
