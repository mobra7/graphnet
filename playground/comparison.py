from graphnet.training.utils import make_dataloader
from graphnet.models import Model
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.detector.icecube import IceCube86
from graphnet.training.labels import Direction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,2,1,0"


model_path = './plots_08_07_2'
freedom = pd.read_pickle(f'{model_path}/performance.pkl')
dynedge_model = Model.load(f'./dynedge_baseline_3/model.pth')

#selection = list(pd.read_pickle(f'{model_path}/performance_events.pkl')['event_no'])
selection = list(freedom['event_no'])

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

dataloader = make_dataloader(
    db=path,
    graph_definition=graph_definition,
    pulsemaps=pulsemap,
    features=features,
    truth=truth,
    batch_size=batch_size,
    num_workers=num_workers,
    truth_table=truth_table,
    selection=selection,
    labels = {'direction': Direction()},
    shuffle = False
)


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

dynedge = dynedge_model.predict_as_dataframe(
    dataloader,
    additional_attributes=additional_attributes,
    prediction_columns=prediction_columns,
    gpus=[1],
)

def angle_dynedge(az_true, zen_true, x_pred, y_pred, z_pred):

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

df = pd.DataFrame()
df['dynedge'] = angle_dynedge(dynedge['azimuth'], dynedge['zenith'], dynedge['dir_x_pred'], dynedge['dir_y_pred'], dynedge['dir_z_pred'])

def angle_model(az_true, zen_true, az_pred, zen_pred):

    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    scalar_prod =  np.clip(scalar_prod, -1, 1)

    return np.abs(np.arccos(scalar_prod))

df['max llh'] = angle_model(freedom['truth_az'], freedom['truth_ze'], freedom['max_llh_az'], freedom['max_llh_ze'])

bin_edges = np.linspace(0, 20000, 21)
bin_centers = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges) - 1)]

df['energy_bin'] = pd.cut(dynedge['energy'], bins=bin_edges)
df['azimuth'] = pd.cut(dynedge['azimuth'], bins = bin_edges)

percentiles = [16,50,84]

def calc_percentiles(group):
    if len(group) == 0:
        return pd.Series([np.nan] * len(percentiles), index=[f'percentile_{p}' for p in percentiles])
    return pd.Series(np.degrees(np.percentile(group, percentiles)), index=[f'percentile_{p}' for p in percentiles])

def plot_percentiles_comparison(df, columns, title, filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = list(mcolors.TABLEAU_COLORS.values())
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # Different line styles for percentiles

    for i, column in enumerate(columns):
        base_color = colors[i % len(colors)]  # Cycle through colors if more columns than colors

        percentiles_df = df.groupby('energy_bin')[column].apply(calc_percentiles).reset_index()
        percentiles_df = percentiles_df.pivot(index='energy_bin', columns='level_1', values=column).reset_index()

        for j, p in enumerate(percentiles):
            ax1.plot(bin_centers, 
                     percentiles_df[f'percentile_{p}'],
                     linestyle=linestyles[j],
                     linewidth=2, 
                     color=base_color, 
                     label=f'{column} {p}th Percentile')



    ax1.set_xlabel('energy')
    ax1.set_ylabel('Opening Angle [deg]')
    ax1.set_title(title)
    ax1.set_ylim(0, 6)
    ax1.set_xlim(0, bin_edges[-1])
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = ax1.twinx()
    hist_data, hist_bins, _ = ax2.hist(dynedge['energy'], bins=bin_edges, alpha=0.3, color='gray', edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')

    ax1.set_xticks(bin_edges[::2])
    ax1.set_xticklabels([int(edge) for edge in bin_edges[::2]])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Call the function with both columns
plot_percentiles_comparison(df, ['dynedge', 'max llh'], 'Comparison of Dynedge and SBI Method', f'{model_path}/comparison.pdf')

print('model mean opening angle: ', np.degrees(np.mean(df['max llh'])))
print('baseline mean opening angle: ', np.degrees(np.mean(df['dynedge'])))