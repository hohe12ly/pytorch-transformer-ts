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

# %%
# DEBUG: print gluonts dataset metadata
for name in ['m4_weekly', 'traffic'] + TRAIN_DATASET_NAMES:
    t_dataset = get_dataset(name, path=dataset_path)
    print('=====', 'dataset:', name, '=====')
    print('freq:', t_dataset.metadata.freq, 'prediction_length:', t_dataset.metadata.prediction_length)
    for x in t_dataset.train:
        ts_shape_train = x['target'].shape
        break
    print('train:', 'type', type(t_dataset.train), 'num_series', len(t_dataset.train), 'ts shape', ts_shape_train)
    for x in t_dataset.test:
        ts_shape_test = x['target'].shape
        break
    print('test:', 'type', type(t_dataset.test), 'num_series', len(t_dataset.test), 'ts shape', ts_shape_test)

# %%
# PSML datasets definition
psml_raw_dataset_names = ['psml_pmu0_1min', 'psml_pmu1_1min', 'psml_pmu2_1min']
psml_23bus_ts_targets = ['Vm', 'Va'] # Voltage magnitude, Voltage angle
psml_dataset_names = [dname + '_' + ts_target for dname in psml_raw_dataset_names for ts_target in psml_23bus_ts_targets]

# bus IDs list. List index is also graph node index
psml_23bus_graph_nodes = ['101', '102', '151', '152', '153', '154', '201', '202', '203', '204', '205', '206', '211', '3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3011', '3018']

psml_raw_datasets = {
    'psml_pmu0_1min': {
        'freq': '1T',
        'start': '2018-01-01 00:00:00',
        'num_series': len(psml_23bus_graph_nodes),
        'prediction_length': 60, # 1 hour
        'rolling_evaluations': 6, # 6 hours
        'num_time_steps': 532_374,
        'ts_targets': psml_23bus_ts_targets,
        'graph_nodes': psml_23bus_graph_nodes,
        'raw_data_path': '/mnt/data/home/yxl/data/PSML/minute-pmu/case-0/pf_result_0.csv'
    }, 
    'psml_pmu1_1min': {
        'freq': '1T',
        'start': '2019-01-01 00:00:00',
        'num_series': len(psml_23bus_graph_nodes),
        'prediction_length': 60, # 1 hour
        'rolling_evaluations': 6, # 6 hours
        'num_time_steps': 524_161,
        'ts_targets': psml_23bus_ts_targets,
        'graph_nodes': psml_23bus_graph_nodes,
        'raw_data_path': '/mnt/data/home/yxl/data/PSML/minute-pmu/case-1/pf_result_1.csv'
    }, 
    'psml_pmu2_1min': {
        'freq': '1T',
        'start': '2019-01-01 00:00:00',
        'num_series': len(psml_23bus_graph_nodes),
        'prediction_length': 60, # 1 hour
        'rolling_evaluations': 6, # 6 hours
        'num_time_steps': 524_161,
        'ts_targets': psml_23bus_ts_targets,
        'graph_nodes': psml_23bus_graph_nodes,
        'raw_data_path': '/mnt/data/home/yxl/data/PSML/minute-pmu/case-2/pf_result_2.csv'
    }, 
}

psml_num_time_steps = 100_000
psml_train_ratio = 0.8

# %%
print(psml_dataset_names)
# %%
default_dataset_writer = JsonLinesWriter()

#%%
# generate psml_pmu0_1min_vm and psml_pmu0_1min_va GluonTS datasets.
# each has 23 bus time series, ignoring edge/branch time series for now.
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

for raw_dataset_name in psml_raw_dataset_names:
    print('processing raw_dataset:', raw_dataset_name, '...')
    print('=====================================')
    raw_dataset = psml_raw_datasets[raw_dataset_name]
    #df_raw = pd.read_csv(raw_dataset['raw_data_path'], skipinitialspace=True, index_col=0, parse_dates=True, nrows=psml_num_time_steps)
    df_raw = pd.read_csv(raw_dataset['raw_data_path'], sep='\s*,\s*', index_col=0, parse_dates=True, nrows=psml_num_time_steps)

    # get timeseries targets
    for ts_target in raw_dataset['ts_targets']:
        print('processing ts_target:', ts_target, '...')
        print('-------------------------------------')
        # construct time_index
        time_index = pd.period_range(
            start=raw_dataset['start'],
            freq=raw_dataset['freq'],
            periods=psml_num_time_steps,
        )
        print('time_index:', 'start=', time_index[0], 'freq=', time_index.freqstr, 'periods=', time_index.shape[0])

        # check time index consistency
        if time_index.shape[0] != df_raw.shape[0]:
            print('ERROR: time index shape mismatch.')
            continue
        if not time_index[0].to_timestamp() == df_raw.index[0]:
            print('WARN: time index broken at start. Ignored.')
        if not time_index[-1].to_timestamp() == df_raw.index[-1]:
            print('WARN: time index broken. Checking but will be ignored.')
            check_timeindex_inconsistency(time_index, df_raw)
            
        # extract ts_target columns from raw dataset
        cols = [ ts_target + '_' + node for node in raw_dataset['graph_nodes'] ]
        df_ts_target = df_raw[cols]
        # check NaN values
        if df_ts_target.isnull().values.any():
            print('ERROR: NaN values in ts_target=', ts_target)
            continue
        print('num_series:', len(cols))
        print('series columns:', cols)

        # get training part
        num_train = int(psml_num_time_steps * psml_train_ratio)
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
            ts = df_ts_target[ts_target + '_' + bus_id]
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
                    "feat_static_cat": [bus_id],
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
                        "feat_static_cat": [bus_id],
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
        rootpath = "/mnt/data/home/yxl/test/test_ai/datasets"
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
print(time_index[0], time_index[-1], time_index.shape)
#print(df_raw['time'].iloc[0], df_raw['time'].iloc[-1], df_raw.shape)
print(df_raw.index[0], df_raw.index[-1], df_raw.shape)
df_ts_target.head()

# %%
ts[time_index[0]:time_index[5]]

# %%
for col in cols[:5]:
    df_ts_target[col].plot()
plt.legend()
plt.show()

# %%
print(rootpath)
print('"', str(dataset_rootpath), '"')
newpath = dataset_rootpath / 'test'
print(type(newpath))
# %%
# DEBUG: load dataset test
print('the last dataset:', dataset_path)
d = load_datasets(
        metadata=dataset_path,
        train=dataset_path / "train",
        test=dataset_path / "test",
    )
# %%
d.metadata
# %%
for x in d.train:
    print(x['target'].shape, x['start'], x['feat_static_cat'], x['item_id'])
# %%
for x in d.test:
    print(x['target'].shape, x['start'], x['feat_static_cat'], x['item_id'])
# %%
