## training and testing on PSML Millisecond Forced Oscillation dataset
# %%
import argparse
import json
import random
import numpy as np
import pandas as pd

# YL
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, LearningRateFinder, LearningRateMonitor

from pathlib import Path
import pathlib
from glob import glob
from hashlib import sha1
import pytorch_lightning as pl


from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.evaluation._base import aggregate_valid
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset, load_datasets # YL
from gluonts.transform import ValidationSplitSampler

from estimator import LagLlamaEstimator
import os

import matplotlib.pyplot as plt

#dataset_path = Path("/home/toolkit/datasets")
dataset_path = Path("/mnt/data/home/yxl/test/test_ai/datasets")

TRAIN_DATASET_NAMES = [
    "airpassengers",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "electricity",
    "electricity_weekly",
    "exchange_rate",
    "fred_md",
    "hospital",
    "kaggle_web_traffic_weekly",
    "kdd_cup_2018_without_missing",
    "london_smart_meters_without_missing",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "pedestrian_counts",
    "rideshare_without_missing",
    "saugeenday",
    "solar-energy",
    "solar_10_minutes",
    "solar_weekly",
    "taxi_30min",
    "temperature_rain_without_missing",
    "tourism_monthly",
    "uber_tlc_daily",
    "uber_tlc_hourly",
    "vehicle_trips_without_missing",
    "weather",
    "wiki-rolling_nips",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
    "wind_farms_without_missing",
]

# %%
class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])

def create_sliding_window_dataset(name, window_size, is_train=True):
    # Splits each time series into non-overlapping sliding windows
    global_id = 0

    freq = get_dataset(name, path=dataset_path).metadata.freq
    data = ListDataset([], freq=freq)
    dataset = get_dataset(name, path=dataset_path).train if is_train else get_dataset(name, path=dataset_path).test

    for x in dataset:
        windows = []
        for i in range(0, len(x['target']), window_size):
            windows.append({
                'target': x['target'][i:i+window_size],
                'start': x['start'] + i,
                'item_id': str(global_id),
                'feat_static_cat': np.array([0]),
            })
            global_id += 1
        data += ListDataset(windows, freq=freq)
    return data

def create_test_dataset(name, window_size):
    # Similar to `create_sliding_window_dataset` but for test dataset
    dataset = get_dataset(name, path=dataset_path)
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length

    data = []
    for x in dataset.test:
        offset = len(x['target']) - window_size - prediction_length
        if offset > 0:
            target = x['target'][-(window_size + prediction_length):]
            data.append({
                'target': target,
                'start': x['start'] + offset,
            })
        else:
            data.append(x)
    return ListDataset(data, freq=freq), prediction_length

# YL start
# PSML Millisecond Forced Oscillation dataset
psml_milli_dataset_names = [ os.path.basename(p) for p in glob(str(dataset_path / "Forced_Oscillation_*")) ]

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

column_map = {
    'trans': data_column_map_trans,
    'dist': data_column_map_dist,
}
datacolumn_list = {
    'trans': list(data_column_map_trans.values())[1:], # skip 'Time'
    'dist': list(data_column_map_dist.values())[1:], # skip 'Time'
}

def create_psml_sliding_window_dataset(name, window_size, is_train=True):
    psml_dataset_path = dataset_path / name
    d = load_datasets(
        metadata=psml_dataset_path,
        train=psml_dataset_path / "train",
        test=psml_dataset_path / "test",
    )
    global_id = 0
    data = ListDataset([], freq=d.metadata.freq)
    dataset = d.train if is_train else d.test
    for x in dataset:
        windows = []
        for i in range(0, len(x['target']), window_size):
            windows.append({
                'target': x['target'][i:i+window_size],
                'start': x['start'] + i,
                'item_id': str(global_id),
                'feat_static_cat': np.array([0]),
            })
            global_id += 1
        data += ListDataset(windows, freq=d.metadata.freq)
    return data

def create_psml_test_dataset(name, window_size):
    # Similar to `create_sliding_window_dataset` but for test dataset
    psml_dataset_path = dataset_path / name
    dataset = load_datasets(
        metadata=psml_dataset_path,
        train=psml_dataset_path / "train",
        test=psml_dataset_path / "test",
    )
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length

    data = []
    for x in dataset.test:
        offset = len(x['target']) - window_size - prediction_length
        if offset > 0:
            target = x['target'][-(window_size + prediction_length):]
            data.append({
                'target': target,
                'start': x['start'] + offset,
            })
        else:
            data.append(x)
    return ListDataset(data, freq=freq), prediction_length

def plot_pred_results(forecasts, tss, name, odir, nodelist, num_nodes, prediction_length, rolling_eval_index, num_rolling_evals, nodedim_is_first=True):
    '''
    plot forecast and ground truth as well quantile info on all graph nodes at the `rolling_eval_index`th 
    rolling evaluation with `prediction_width`. `forecasts` and `tss` is a linear list of time series data.
    the index i is determined in the order of `[node_index, rolling_eval_index]` or 
    `[rolling_eval_index, node_index]`, depending on `nodedim_is_first`. 

    unfortunately, info about `num_rolling_evals` and `num_nodes` is not stored in the dataset metadata. 
    so this function is dataset specific. you have to know the dataset structure to use this function.
    '''
    # psml test data order: (node/bus, rolling eval): timesteps data
    #num_buses = 23
    #num_rolling_evals = 6
    #prediction_length, num_buses, num_rolling_evals

    fig, axes = plt.subplots(num_nodes, 1, figsize=(10, num_nodes * 5))
    data_index = [ (i * num_rolling_evals + rolling_eval_index) if nodedim_is_first else (i + rolling_eval_index * num_nodes) for i in range(num_nodes) ]
    
    #print('shape of each tss: ', [ (i, data_index[i], tss[data_index[i]].shape) for i in range(num_nodes) ] )
    
    for i, ax in enumerate(axes): # i is node index
        ax.plot(tss[data_index[i]][-(prediction_length * 2):].to_timestamp())
        plt.sca(ax)
        forecasts[data_index[i]].plot(intervals=(0.5, 0.8, 0.9, 0.95), color='m')
        plt.legend(['ground truth', 'pred mean', '0.5', '0.8', '0.9', '0.95'])
        plt.title('node: ' + str(i) + ' , node_name: ' + nodelist[i])
    plt.savefig(odir + '/' + 'perf_pred_' + name + '_' +
                '_numnodes_' + str(len(nodelist)) + 
                '_predlen_' + str(prediction_length) + 
                '_at_rollingeval_' + str(rolling_eval_index) +
                '.png')

    fig, axes = plt.subplots(num_nodes, 1, figsize=(10, num_nodes * 5))
    for i, ax in enumerate(axes):
        ax.plot(tss[data_index[i]].to_timestamp())
        plt.sca(ax)
        plt.title('node: ' + str(i) + ' , node_name: ' + nodelist[i] + ' [timesteps: ' + str(len(tss[data_index[i]])) + ']')
    plt.savefig(odir + '/' + 'testdata_allbuses_' + name + '_' +
                '_numnodes_' + str(len(nodelist)) + 
                '_predlen_' + str(prediction_length) + 
                '_at_rollingeval_' + str(rolling_eval_index) +
                '.png')
    
    # save data
    np.save(
        odir + '/' + 'testdata_allbuses_' + name + '_' +
        '_numnodes_' + str(len(nodelist)) + 
        '_predlen_' + str(prediction_length) + 
        '_at_rollingeval_' + str(rolling_eval_index) +
        '.npy'
        ,
       np.array([ tss[data_index[i]].values.reshape(-1) for i in range(num_nodes) ]) # shape of tss[i].values: (context_length + prediction_length,)
    )
    np.save(
        odir + '/' + 'forecastdata_allbuses_' + name + '_' +
        '_numnodes_' + str(len(nodelist)) + 
        '_predlen_' + str(prediction_length) + 
        '_at_rollingeval_' + str(rolling_eval_index) +
        '.npy'
        ,
        np.array([ forecasts[data_index[i]].samples for i in range(num_nodes) ]) # shape of forecasts[i].samples: (num_samples, prediction_length)
    )
     
# YL end

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    experiment_name = f'data-scaling-context-{args.context_length}-layer-{args.n_layer}-n_embd-{args.n_embd}-n_head-{args.n_head}-aug-{args.aug_prob}-{args.aug_rate}'
    fulldir = os.path.join(pathlib.Path(__file__).parent.resolve(), "model-size-scaling-logs"+"."+args.jobname, str(args.seed)) # Always creates the experiment directory inside "lag-llama"
    os.makedirs(fulldir, exist_ok=True)
    fulldir_experiments = os.path.join(fulldir, "experiments")
    os.makedirs(fulldir_experiments, exist_ok=True)

    # Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
    lightning_version_to_use, ckpt_path = None, None
    max_epoch = -1
    if "lightning_logs" in os.listdir(fulldir_experiments):
        for lightning_version in os.listdir(fulldir_experiments+"/lightning_logs/"):
            ckpts = glob(fulldir_experiments+"/lightning_logs/" + lightning_version + "/checkpoints/*.ckpt")
            #if len(ckpts): 
            for ckpt in ckpts:
                #epoch = int(ckpts[0][ckpts[0].find("=")+1:ckpts[0].find("-step")])
                epoch = int(ckpt[ckpt.find("=")+1:ckpt.find("-step")])
                if epoch > max_epoch:
                    lightning_version_to_use = lightning_version
                    max_epoch = epoch
                    #ckpt_path = ckpts[0]
                    ckpt_path = ckpt
        if lightning_version_to_use: print("Using lightning_version", lightning_version_to_use, "with epoch", max_epoch, "restoring from checkpoint at path", ckpt_path)
    else: print ("no lightning logs found. Training from scratch.")
    
    logger = CSVLogger(
        save_dir=fulldir_experiments,
        flush_logs_every_n_steps=1,
        version=lightning_version_to_use
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(args.early_stopping_patience), verbose=True, mode="min")
    model_checkpointing = ModelCheckpoint(save_top_k=1)
    callbacks = [early_stop_callback, model_checkpointing]

    estimator = LagLlamaEstimator(
        prediction_length=1,
        context_length=args.context_length,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        scaling="std",
        aug_prob=args.aug_prob,
        aug_rate=args.aug_rate,
        num_batches_per_epoch=args.num_batches_per_epoch,
        ckpt_path=ckpt_path,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            #devices=[args.gpu],
            devices=[0],
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=fulldir_experiments
        ),
    )
    
    num_parameters = sum(p.numel() for p in estimator.create_lightning_module().parameters())
    print("num_parameters : ", num_parameters)

    window_size = estimator.context_length + max(estimator.lags_seq) + estimator.prediction_length
    # Here we make a window slightly bigger so that instance sampler can sample from each window
    # An alternative is to have exact size and use different instance sampler (e.g. ValidationSplitSampler)
    window_size = 10 * window_size

    # We change ValidationSplitSampler to add min_past
    estimator.validation_sampler = ValidationSplitSampler(
        min_past=estimator.context_length + max(estimator.lags_seq),
        min_future=estimator.prediction_length,
    )

    if args.test:
        print('Testing only')
    else:
        # Create training data
        train_data, val_data = [], []
        # YL start
        for name in psml_milli_dataset_names[:480]: # case 0, case 1 for training
            new_data = create_psml_sliding_window_dataset(name, window_size)
            train_data.append(new_data)

            new_data = create_psml_sliding_window_dataset(name, window_size, is_train=False)
            val_data.append(new_data)
        # YL end
        for name in TRAIN_DATASET_NAMES:
            new_data = create_sliding_window_dataset(name, window_size)
            train_data.append(new_data)

            new_data = create_sliding_window_dataset(name, window_size, is_train=False)
            val_data.append(new_data)
        # Here weights are proportional to the number of time series (=sliding windows)
        weights = [len(x) for x in train_data]
        # Here weights are proportinal to the number of individual points in all time series
        # weights = [sum([len(x["target"]) for x in d]) for d in train_data]

        train_data = CombinedDataset(train_data, weights=weights)
        val_data = CombinedDataset(val_data, weights=weights)

        # Train
        # TODO: Depending on the stopping criterion, saved checkpoint will be based on validation
        # and the test set for these datasets will be the same (doesn't impact zero-shot experiment)
        train_output = estimator.train_model(
            training_data=train_data,
            validation_data=val_data,
            ckpt_path=ckpt_path
        )
    if not args.test:
        estimator.ckpt_path = train_output.trainer.checkpoint_callback.best_model_path
    print(f'Use checkpoint: {estimator.ckpt_path}')

    # for name in ['m4_weekly', 'traffic'] + TRAIN_DATASET_NAMES:
    for name in psml_milli_dataset_names[-20:] + ['m4_weekly', 'traffic']: # + TRAIN_DATASET_NAMES[0:5]:
        print(f'Predict on {name}')
        if name in psml_milli_dataset_names[-20:]:
            test_data, prediction_length = create_psml_test_dataset(name, window_size)
        else:
            test_data, prediction_length = create_test_dataset(name, window_size)
        print(f'{name} prediction length: {prediction_length}')

        # Adapt evaluator to new dataset
        estimator.prediction_length = prediction_length
        estimator.batch_size = max(30 // estimator.prediction_length, 1) # Some heuristic for GPU memory (TODO: change)
        predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module(),
        )
        # Make evaluations
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=predictor,
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        if name in psml_milli_dataset_names[-20:]:
            cols = datacolumn_list[name.split('_')[3]]
            plot_pred_results(forecasts, tss, name, logger.log_dir, cols, len(cols), prediction_length, 0, 5)

        evaluator = Evaluator(num_workers=1, aggregation_strategy=aggregate_valid)
        agg_metrics, item_metrics = evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        agg_metrics["trainable_parameters"] = num_parameters
        agg_metrics["n_layer"] = args.n_layer
        agg_metrics["n_embd"] = args.n_embd
        agg_metrics["n_head"] = args.n_head

        print("logger.log_dir : ", logger.log_dir)
        print("os.path.exists(logger.log_dir) : ", os.path.exists(logger.log_dir))

        if not os.path.exists(logger.log_dir):
            os.makedirs(logger.log_dir)
        with open(f'{logger.log_dir}/{name}.json', 'w') as f:
            json.dump(agg_metrics, f)

        item_metrics.to_csv(f'{logger.log_dir}/{name}_item_metrics.csv')

if __name__ == '__main__':
    print("YL training on gpus:", torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    #estimator args
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--jobname", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--aug_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    # estimator trainer kwarg args
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1000)
    #parser.add_argument("--gpu", type=int, default=0)
    # Other args
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--early_stopping_patience', default=50)
    # Model
    args = parser.parse_args()
    print("YL arguments:")
    print(args)

    train(args)
