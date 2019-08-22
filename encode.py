import pandas as pd
import csv

feature_cols = []
class_cols = []
total_data = []

names = ['protocol', 'range(m)', 'power_src', 'weight(g)', 'processing_power(ghz)', 'device_type'] # column headings
file = pd.read_csv('dataset.csv', names=names)

non_enc = pd.DataFrame(file, columns=['range(m)', 'weight(g)', 'processing_power(ghz)', 'device_type']) # columns that do not require encoding
enc_cols = pd.DataFrame(file, columns=['protocol', 'power_src']) # columns that require Dummy encoding
new_cols = pd.get_dummies(enc_cols, columns=enc_cols) # encode 'protocol' and 'power_src' columns

for col in new_cols:
    new_cols[col] = new_cols[col].astype(object) # change the type of newly encoded columns to object
    concat_datasets = pd.concat([non_enc, new_cols], axis=1) # concatenate the encoded columns with non-encoded columns

    cols = list(concat_datasets.columns.values)
    cols.pop(cols.index('device_type')) # pop the column 'type' from the list of columns
    new_dataset = concat_datasets[cols+['device_type']] # append 'type' to end of list
