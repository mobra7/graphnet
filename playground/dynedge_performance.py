from graphnet.models import Model
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.training.labels import Direction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"

path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_2.db"
pulsemap = 'InIcePulses'
target = 'scrambled_class'
truth_table = 'truth'
gpus = [1]
max_epochs = 30
early_stopping_patience = 5
batch_size = 500
num_workers = 30
wandb =  False

features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86

graph_definition = KNNGraph(detector=IceCube86())

(
    training_dataloader,
    validation_dataloader,
) = make_train_validation_dataloader(
    db=path,
    graph_definition=graph_definition,
    pulsemaps=pulsemap,
    features=features,
    truth=truth,
    batch_size=batch_size,
    num_workers=num_workers,
    truth_table=truth_table,
    selection=None,
    test_size = 0.3,
    labels = {'direction': Direction()}
)

model = Model.load('/scratch/users/mbranden/graphnet/playground/dynedge_baseline/model.pth')


additional_attributes = [
    "zenith",
    "azimuth",
    "event_no",
    "energy"
]
prediction_columns = [
    "dir_x_pred",
    "dir_y_pred",
    "dir_z_pred",
    "dir_kappa_pred",
]

assert isinstance(additional_attributes, list) 

df = model.predict_as_dataframe(
    validation_dataloader,
    additional_attributes=additional_attributes,
    prediction_columns=prediction_columns,
    gpus=[1],
)

def angle(az_true, zen_true, x_pred, y_pred, z_pred):

    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    zen_pred = np.arccos(z_pred)

    sa2 = y_pred/np.sin(zen_pred)
    ca2 = x_pred/np.sin(zen_pred)
    sz2 = np.sin(zen_pred)
    cz2 = z_pred
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    scalar_prod =  np.clip(scalar_prod, -1, 1)

    return np.abs(np.arccos(scalar_prod))

df['dynedge-truth'] = angle(df['azimuth'], df['zenith'], df['dir_x_pred'], df['dir_y_pred'], df['dir_z_pred'])

bin_edges = np.linspace(0, 20000, 21)
bin_centers = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges) - 1)]

df['energy_bin'] = pd.cut(df['energy'], bins=bin_edges)

percentiles = [16, 50, 84]

def calc_percentiles(group):
    if len(group) == 0:
        return pd.Series([np.nan] * len(percentiles), index=[f'percentile_{p}' for p in percentiles])
    return pd.Series(np.degrees(np.percentile(group, percentiles)), index=[f'percentile_{p}' for p in percentiles])

def plot_percentiles(df, column, title, filename):
    percentile_df = df.groupby('energy_bin')[column].apply(calc_percentiles).reset_index()
    percentile_df = percentile_df.pivot(index='energy_bin', columns='level_1', values=column).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for p in percentiles:
        ax1.plot(bin_centers, percentile_df[f'percentile_{p}'], marker='o', label=f'{p}th Percentile')

    ax1.set_xlabel('Energy')
    ax1.set_ylabel('Opening Angle [deg]')
    ax1.set_title(title)
    ax1.set_ylim(0,6)
    ax1.set_xlim(0,bin_edges[-1])
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = ax1.twinx()
    hist_data, hist_bins, _ = ax2.hist(df['energy'], bins=bin_edges, alpha=0.3, color='gray', edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')

    ax1.set_xticks(bin_edges[::2])
    ax1.set_xticklabels([int(edge) for edge in bin_edges[::2]])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_percentiles(df, 'dynedge-truth', 'DynEdge Baseline to Truth', './dynedge_baseline/performance.pdf')
df.to_pickle('./dynedge_baseline/performance.pkl')
print(np.degrees(np.mean(df['dynedge-truth'])))
