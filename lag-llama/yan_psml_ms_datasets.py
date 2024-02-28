
## This script is used to generate GluonTS datasets from the PSML Millisecond dataset.
# level 1: Forced Oscillation + Natural Oscillation
# level 2: Scenarios (row_###)
# level 3: trans + dist time series. The paper says: "Each time-series has 
#          a sequence length of 960 observations, representing 4 seconds in the system 
#          recorded at 240 Hz. There are 91 dimensions for each time-series, including 
#          voltage, current and power measurements across the transmission system."
# In data csv, the time field starts with 0.0, not a timestamp. The increment is not
# constant, roughly at (4 seconds / 960 records) = 4.1667 ms. Thus, freq='4167U', microsecond level.
# The actual start time is in `info.csv`. 
# Issues:
#    The `dist.csv` in Natural Oscillation is incorrect: each timestep occurs twice.
#    Forced_Oscillation/row_149/info.csv: type has this weird invalid csv format: type, ['1', '2', '3']. ',' is removed.
#    Natural_Oscillation/row_328: info.csv does not exist.
#    Natural_Oscillation/row_174: trans.csv column order wrong: VARS 3008 TO 3018 CKT '1 ', Time(s), ...

# %%
import json
import random
import numpy as np
import pandas as pd

from typing import List, NamedTuple, Optional, cast

from pathlib import Path
import pathlib

from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import MetaData, TrainDatasets, load_datasets
from gluonts.dataset.repository._util import metadata
from gluonts.dataset.jsonl import JsonLinesWriter

import matplotlib.pyplot as plt

# from gluonts.dataset.pandas import PandasDataset
# from gluonts.dataset.common import ListDataset

import glob
import os

def generate_column_map():
    sample_dir = '/mnt/data/home/yxl/data/PSML/milli-pmu/Forced_Oscillation/row_2'
    sample_trans = sample_dir + '/trans.csv'
    df = pd.read_csv(sample_trans, sep='\s*,\s*',)
    cols = df.columns.values
    column_map = {} # for trans.csv
    for col in cols:
        col1 = col.strip()
        if col1.startswith('Time(s)'):
            column_map[col] = 'Time'
        elif col1.startswith('VOLT '):
            column_map[col] = 'Bus_' + col1.split(' ')[1] # ' VOLT 205 [SUB230 230.00]'
        elif col1.startswith('POWR ') or col1.startswith('VARS '):
            strs = col1.split(' ') # " VARS 101 TO 151 CKT '1 '", " POWR 102 TO 151 CKT '1 '"
            edge_index = strs[5].split("'")[1] # '1, '2
            vtype = 'P' # POWR, active power voltage
            if strs[0].startswith('VARS'):
                vtype = 'Q' # VARS, reactive power voltage
            column_map[col] = vtype + '_' + strs[1] + '_' + strs[3] + '_' + edge_index
        else:
            raise ValueError('Unknown column name pattern:', col1)
    sample_dist = sample_dir + '/dist.csv'
    df = pd.read_csv(sample_dist, sep='\s*,\s*',)
    cols = df.columns.values
    dist_column_map = {} # for dist.csv
    for col in cols:
        col1 = col.strip()
        if col1.startswith('Time(s)'):
            dist_column_map[col] = 'Time'
        else:
            dist_column_map[col] = col1
    return column_map, dist_column_map

data_column_map_trans, data_column_map_dist = generate_column_map() 

# %%
column_map = {
    'trans': data_column_map_trans,
    'dist': data_column_map_dist,
}
datacolumn_list = {
    'trans': list(data_column_map_trans.values())[1:], # skip 'Time'
    'dist': list(data_column_map_dist.values())[1:], # skip 'Time'
}

# %%
# time series attributes of the PSML Millisecond dataset
data_freq = '4167U' # microsecond level
data_num_time_steps = 960
#  datasets definition
data_raw_dataset_names = []
data_raw_datasets = {}
data_ts_targets = ['Voltage'] # Voltage 
data_root_path = '/mnt/data/home/yxl/data/PSML/milli-pmu'
for osci in ['Forced_Oscillation', 'Natural_Oscillation']:
    for scenario in glob.glob(data_root_path + '/' + osci + '/row_*'):
        scenario_id = int(os.path.basename(scenario).split('_')[1])
        #print('identifying data:', osci, ', scenario_id:', scenario_id)
        # get start time from info.csv
        info_csv = scenario + '/info.csv'
        if os.path.exists(info_csv):
            df_info = pd.read_csv(info_csv, header=None)
            str_start_time = df_info.transpose().iloc[1].values[0] # contains leading/trailing spaces, but pd can handle it
        else:
            str_start_time = '2018-01-01 00:00:00' # give it something
        # get trans.csv data
        for data_type in ['trans', 'dist']:
            name = osci + '_' + str(scenario_id) + '_' + data_type
            data_raw_dataset_names.append(name)
            data_raw_datasets[name] = {
                'freq': data_freq,
                'start': str_start_time,
                'num_series': len(datacolumn_list[data_type]),
                'prediction_length': 60, # 0.0041 * 60 = 0.246 seconds
                'rolling_evaluations': 5, # up to 0.246 * 5 = 1.23 seconds
                'num_time_steps': data_num_time_steps,
                'ts_targets': data_ts_targets,
                'graph_nodes': datacolumn_list[data_type], # TODO: separate nodes and edges columns
                'column_names': {
                    'Voltage': datacolumn_list[data_type],
                },
                'raw_data_path': data_root_path + '/' + osci + '/row_' + str(scenario_id) + '/' + data_type + '.csv'
        }

# %%
data_dataset_names = [dname + '_' + ts_target for dname in data_raw_dataset_names for ts_target in data_ts_targets]
print('num_raw_data_files:', len(data_raw_dataset_names))
data_train_ratio = 0.8

# %%
print(data_dataset_names)
# %%
default_dataset_writer = JsonLinesWriter()

#%%
# generate PSML Millisecond PMU GluonTS datasets.
# TODO: handle node columns and edge columns for graph data gen.
# the raw dataset is loaded from CSV file as Pandas DataFrame.
# Rows are timestamps, columns are bus IDs.

def check_timeindex_inconsistency(time_index, df_raw):
    timestamps = time_index.to_timestamp()
    for i in range(timestamps.shape[0]):
        if timestamps[i] != df_raw.index[i]:
            print('time index broken at:', i, timestamps[i], df_raw.index[i])
            print('trace:')
            for ii in range(i-3, i+3):
                print(ii, timestamps[ii], df_raw.index[ii])
            break

rootpath = "/mnt/data/home/yxl/test/test_ai/datasets"

for raw_dataset_name in data_raw_dataset_names:
    print('=====================================')
    print('processing raw_dataset:', raw_dataset_name, '...')
    
    raw_dataset = data_raw_datasets[raw_dataset_name]
    #df_raw = pd.read_csv(raw_dataset['raw_data_path'], skipinitialspace=True, index_col=0, parse_dates=True, nrows=data_num_time_steps)
    df_raw = pd.read_csv(raw_dataset['raw_data_path'], sep='\s*,\s*', nrows=data_num_time_steps) # index_col=0, parse_dates=True
    df_raw.rename(columns=column_map[raw_dataset_name.split('_')[-1]], inplace=True)
    #print(column_map[raw_dataset_name.split('_')[-1]])
    #print(df_raw.columns)

    # get timeseries targets
    for ts_target in raw_dataset['ts_targets']:
        print('-------------------------------------')
        print('processing ts_target:', ts_target, '...')
        if os.path.exists(rootpath + '/' + raw_dataset_name + '_' + ts_target):
            print('dataset already exists. Ignored.')
            continue
        
        # construct time_index
        time_index = pd.period_range(
            start=raw_dataset['start'],
            freq=raw_dataset['freq'],
            periods=data_num_time_steps,
        )
        print('time_index:', 'start=', time_index[0], 'freq=', time_index.freqstr, 'periods=', time_index.shape[0])

        # check time index consistency
        if time_index.shape[0] != df_raw.shape[0]:
            print('ERROR: time index shape mismatch.')
            continue
        ## PSML Millisecond dataset does not have a constant time increment. so skipping the following checks.
        # if not time_index[0].to_timestamp() == df_raw.index[0]:
        #     print('WARN: time index broken at start. Ignored.')
        # if not time_index[-1].to_timestamp() == df_raw.index[-1]:
        #     print('WARN: time index broken. Checking but will be ignored.')
        #     check_timeindex_inconsistency(time_index, df_raw)
            
        # extract ts_target columns from raw dataset
        #cols = [ ts_target + '_' + node for node in raw_dataset['graph_nodes'] ]
        cols = raw_dataset['column_names'][ts_target]
        print('ts_target columns:', cols)
        df_ts_target = df_raw[cols]
        # check NaN values
        if df_ts_target.isnull().values.any():
            print('ERROR: NaN values in ts_target=', ts_target)
            continue
        print('num_series:', len(cols))
        print('series columns:', cols)

        # get training part
        num_train = int(data_num_time_steps * data_train_ratio)
        training_end = time_index[num_train - 1] # inclusive
        print('num_train:', num_train, 'train_range: [', time_index[0], '--', time_index[num_train - 1], ']')

        # time of the first prediction in test set
        prediction_dates = [
            training_end + i * raw_dataset['prediction_length']
            for i in range(raw_dataset['rolling_evaluations'])
        ]

        # construct train time series
        train_timeseries = [] # dims: [node/bus, ts_length]
        test_timeseries = [] # dims: [node/bus, rolling_eval, ts_length]
        for cat, bus_id in enumerate(raw_dataset['graph_nodes']):
            #ts = df_ts_target[ts_target + '_' + bus_id]
            ts = df_ts_target[cols[cat]]
            ts = pd.Series(ts.values, index=time_index) # reindex with time_index

            # train time series
            ts_train = ts[:time_index[num_train - 1]] # range in time-indexed Series is inclusive [t_start, t_end]
            if len(ts) == 0:
                print('WARN: empty time series for bus_id=', bus_id, '. Ignored.')
                continue
            train_timeseries.append(
                {
                    "target": ts_train.values,
                    "start": ts_train.index[0],
                    "feat_static_cat": [cat], # bus_id is str, this has to be int. TODO: make more static cat vars to embed graph node/eddge info.
                    "item_id": cat,
                }
            )
            if cat == 0:
                print('each train time series shape:', ts_train.shape)

            # test time series
            for rolling_eval, prediction_start_date in enumerate(prediction_dates):
                # print(prediction_start_date)
                prediction_end_date = (
                    prediction_start_date + raw_dataset['prediction_length']
                )
                ts_test = ts[:prediction_end_date]
                test_timeseries.append(
                    {
                        "target": ts_test.values,
                        "start": ts_test.index[0],
                        "feat_static_cat": [cat],
                        "item_id": cat,
                    }
                )
                if cat == 0:
                    print('each test time series shape: [', 'rolling_eval =', rolling_eval, ']:', ts_test.shape)

        assert(len(train_timeseries) == len(raw_dataset['graph_nodes']))
        assert len(test_timeseries) == len(raw_dataset['graph_nodes']) * raw_dataset['rolling_evaluations']

        # construct metadata
        meta = MetaData(
            **metadata(
                cardinality=len(raw_dataset['graph_nodes']),
                freq=raw_dataset['freq'],
                prediction_length=raw_dataset['prediction_length'],
            )
        )

        dataset = TrainDatasets(metadata=meta, train=train_timeseries, test=test_timeseries)

        # save dataset
        dataset_rootpath = Path(rootpath)
        if not dataset_rootpath.exists():
            dataset_rootpath.mkdir(parents=True, exist_ok=True)
        dataset_path = Path(rootpath + '/' + raw_dataset_name + '_' + ts_target)
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
        
        dataset.save(
            path_str=str(dataset_path), writer=default_dataset_writer, overwrite=True
        )
    #     # DEBUG
    #     break
    # # DEBUG
    # break

# %%
# print(time_index[0], time_index[-1], time_index.shape)
# #print(df_raw['time'].iloc[0], df_raw['time'].iloc[-1], df_raw.shape)
# print(df_raw.index[0], df_raw.index[-1], df_raw.shape)
# df_ts_target.head()

# # %%
# ts[time_index[0]:time_index[5]]

# # %%
# for col in cols[:5]:
#     df_ts_target[col].plot()
# plt.legend()
# plt.show()

# # %%
# print(rootpath)
# print('"', str(dataset_rootpath), '"')
# newpath = dataset_rootpath / 'test'
# print(type(newpath))
# %%
# # DEBUG: load dataset test
# print('the last dataset:', dataset_path)
# %%
dataset_path = Path('/mnt/data/home/yxl/test/test_ai/datasets/Forced_Oscillation_99_trans_Voltage')
d = load_datasets(
        metadata=dataset_path,
        train=dataset_path / "train",
        test=dataset_path / "test",
    )
# %%
d.metadata
# %%
d.train
# %%
raw_dataset_name = data_raw_dataset_names[400] # index 90 trnas, raw row_54
raw_dataset = data_raw_datasets[raw_dataset_name]
print(raw_dataset['raw_data_path'])
# %%
df_raw = pd.read_csv(raw_dataset['raw_data_path'], sep='\s*,\s*', nrows=data_num_time_steps) # index_col=0, parse_dates=True
df_raw.rename(columns=column_map[raw_dataset_name.split('_')[-1]], inplace=True)
df_raw.columns
# %%
print(raw_dataset['raw_data_path'])
df_raw.head()
# %%
df_raw.set_index('Time', inplace=True)
df_raw.head()
# %%
df_raw[['Bus_102', 'Bus_151', 'Bus_152', 'Bus_201']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['P_102_151_1', 'P_151_152_1', 'P_151_152_2', 'P_151_201_1']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['Q_102_151_1', 'Q_151_152_1', 'Q_151_152_2', 'Q_151_201_1']].plot(grid=True, xticks=np.arange(0,4,0.1))

# %%
raw_dataset_name = data_raw_dataset_names[401] # index 90 dist
raw_dataset = data_raw_datasets[raw_dataset_name]
print(raw_dataset['raw_data_path'])
df_raw = pd.read_csv(raw_dataset['raw_data_path'], sep='\s*,\s*', nrows=data_num_time_steps) # index_col=0, parse_dates=True
df_raw.rename(columns=column_map[raw_dataset_name.split('_')[-1]], inplace=True)
df_raw.set_index('Time', inplace=True)
print(df_raw.columns)
df_raw.head()
# %%
df_raw[['3005.sourcebus.1', '3005.sourcebus.2', '3005.sourcebus.3']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['3008.650.1', '3008.650.2', '3008.650.3']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['3005.650.1', '3005.650.2', '3005.650.3']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['3008.632.1', '3008.632.2', '3008.632.3']].plot(grid=True, xticks=np.arange(0,4,0.1))
# %%
df_raw[['3008.671.1', '3008.671.2', '3008.671.3']].plot(grid=True, xticks=np.arange(0,4,0.1))


# %%
for i, n in enumerate(data_raw_dataset_names):
    if n == 'Forced_Oscillation_90_dist':
        print(i, n)
        break
# %%
# for x in d.train:
#     print(x['target'].shape, x['start'], x['feat_static_cat'], x['item_id'])
# %%
# for x in d.test:
#     print(x['target'].shape, x['start'], x['feat_static_cat'], x['item_id'])
# %%
# # test pandas PeriodIndex
# import pandas as pd
# time_index = pd.period_range(
#     start='2024-02-29 00:00:00',
#     freq='U',
#     periods=100_000,
# )
# # %%
# for i in [0, 100, 1000, 2000]:
#     ti = time_index[i]
#     print(i, ':', ti.second, ti.minute, ti.hour, ti.day, ti.month, ti.year)
# # %%
# time_index[100].to_timestamp()
# %%
# from pathlib import Path
# dataset_path = Path("/mnt/data/home/yxl/test/test_ai/datasets")
# psml_milli_dataset_names = [ os.path.basename(p) for p in glob.glob(str(dataset_path / "Forced_Oscillation_*")) ]

# def generate_column_map():
#     sample_dir = '/mnt/data/home/yxl/data/PSML/milli-pmu/Forced_Oscillation/row_2'
#     sample_trans = sample_dir + '/trans.csv'
#     df = pd.read_csv(sample_trans, sep='\s*,\s*',)
#     cols = df.columns.values
#     column_map = {} # for trans.csv
#     for col in cols:
#         col1 = col.strip()
#         if col1.startswith('Time(s)'):
#             column_map[col] = 'Time'
#         elif col1.startswith('VOLT '):
#             column_map[col] = 'Bus_' + col1.split(' ')[1] # ' VOLT 205 [SUB230 230.00]'
#         elif col1.startswith('POWR ') or col1.startswith('VARS '):
#             strs = col1.split(' ') # " VARS 101 TO 151 CKT '1 '", " POWR 102 TO 151 CKT '1 '"
#             edge_index = strs[5].split("'")[1] # '1, '2
#             vtype = 'P' # POWR, active power voltage
#             if strs[0].startswith('VARS'):
#                 vtype = 'Q' # VARS, reactive power voltage
#             column_map[col] = vtype + '_' + strs[1] + '_' + strs[3] + '_' + edge_index
#         else:
#             raise ValueError('Unknown column name pattern:', col1)
#     sample_dist = sample_dir + '/dist.csv'
#     df = pd.read_csv(sample_dist, sep='\s*,\s*',)
#     cols = df.columns.values
#     dist_column_map = {} # for dist.csv
#     for col in cols:
#         col1 = col.strip()
#         if col1.startswith('Time(s)'):
#             dist_column_map[col] = 'Time'
#         else:
#             dist_column_map[col] = col1
#     return column_map, dist_column_map

# data_column_map_trans, data_column_map_dist = generate_column_map() 

# column_map = {
#     'trans': data_column_map_trans,
#     'dist': data_column_map_dist,
# }
# datacolumn_list = {
#     'trans': list(data_column_map_trans.values())[1:], # skip 'Time'
#     'dist': list(data_column_map_dist.values())[1:], # skip 'Time'
# }

# %%
# for i in [0,1]:
#     name = psml_milli_dataset_names[i]
#     cols = datacolumn_list[name.split('_')[3]]
#     print(name, ':', cols)
# %%
