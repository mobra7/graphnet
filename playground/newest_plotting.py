from typing import (Any, Callable, Dict, List, Optional, Tuple, Type,
                    Union, cast)
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import torch.nn.functional as F

from graphnet.training.utils import make_dataloader
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCube86




model_path = './vMF_IS_09_19'
model = Model.load(f'{model_path}/model.pth')
df = pd.read_pickle(f'{model_path}/performance.pkl')
select_id = 3
max_llh_ze = df['max_llh_ze'][select_id]
max_llh_az = df['max_llh_az'][select_id]
spline_ze = df['spline_ze'][select_id]
spline_az = df['spline_az'][select_id]
truth_ze = df['truth_ze'][select_id]
truth_az = df['truth_az'][select_id]
event_no = [df['event_no'].to_list()[select_id]]


db_path = "/scratch/users/mbranden/sim_files/no_hlc_dev_northern_tracks_full_part_2.db"
pulsemap = 'InIcePulses'
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
batch_size = 500
num_workers = 30
truth_table = 'truth'
graph_definition = KNNGraph(detector=IceCube86())


dataloader = make_dataloader(
    db=db_path,
    graph_definition=graph_definition,
    pulsemaps=pulsemap,
    features=features,
    truth=truth,
    batch_size=batch_size,
    num_workers=num_workers,
    truth_table=truth_table,
    selection=event_no,
    labels = {},
    shuffle = False
)


def rotate_around_axis(x, y, z, axis, angle):
    """
    Rotate points around a given axis by a specified angle.
    Args:
        x, y, z: Cartesian coordinates of points to rotate
        axis: 'x', 'y', or 'z', the axis around which to rotate
        angle: The rotation angle in radians
    Returns:
        Rotated coordinates (x_rot, y_rot, z_rot)
    """
    if axis == 'z':
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        z_rot = z
    elif axis == 'y':
        x_rot = x * np.cos(angle) + z * np.sin(angle)
        z_rot = -x * np.sin(angle) + z * np.cos(angle)
        y_rot = y
    elif axis == 'x':
        y_rot = y * np.cos(angle) - z * np.sin(angle)
        z_rot = y * np.sin(angle) + z * np.cos(angle)
        x_rot = x
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    return x_rot, y_rot, z_rot


def predict_likelihood(model,zen,azi,delta, data: Union[Data, List[Data]]) -> List[Union[torch.Tensor, Data]]:
    """Forward pass, chaining model components."""
    model.inference()
    model.train(mode=False)

    if isinstance(data, Data):
        data = [data]

    zenith = np.linspace(np.pi/2-delta,np.pi/2+delta,100)
    azimuth = np.linspace(-delta,delta,100)
    ze,az = np.meshgrid(zenith, azimuth)
    zenith = torch.tensor(ze.flatten())
    azimuth = torch.tensor(az.flatten())


    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)


    x, y, z = rotate_around_axis(x, y, z, 'y', (zen-np.pi/2))
    x, y, z = rotate_around_axis(x, y, z, 'z', azi)
    directions = torch.stack((torch.tensor(x), torch.tensor(y), torch.tensor(z)), dim=1)

    npix = directions.shape[0]
    x_list = []
    

    for d in tqdm(data):
        x = model.backbone(d)
        for z in range(x.shape[0]):
            x_list.extend(npix*[x[z]])
    
    
    events_count = int(len(x_list)/npix)
    y_list = [directions for _ in range(events_count)]
    x = torch.stack(x_list)
    y = torch.stack(y_list).reshape(len(x_list),3)

    # Add scrambled target to inputs
    x = torch.cat([x, y], dim=1).float()  # Shape: (num_events * npix, feature_dim + 3)
    
    # Pass both latent vec and scrambled target to discriminator
    x = model._discriminator(x)

    # Pass to task
    task_preds = [task(x) for task in model._tasks]
    events_count = len(data)
    pred_chunk = task_preds[0].chunk(events_count)  # Only takes first task for now
    preds = np.array([event_pred.detach().numpy() for event_pred in pred_chunk])

    return preds, az, ze-np.pi/2




skymap, azimuth, zenith = predict_likelihood(model,truth_ze,truth_az,np.deg2rad(7.5),dataloader)
log_skymap = np.where(skymap > 0.000001, skymap, 0.000001)
log_skymap = np.log(log_skymap[0].reshape(azimuth.shape))
max_log_skymap = np.max(log_skymap)
delta_log_skymap = log_skymap - max_log_skymap

fig, ax = plt.subplots(figsize=(6, 6))

cax = ax.pcolormesh(np.degrees(azimuth), 
                    np.degrees(zenith), 
                    delta_log_skymap, cmap=cm.viridis, 
                    shading='gouraud')

cb = plt.colorbar(cax, ax=ax, orientation='horizontal')
cb.set_label(r'$\Delta$log-likelihood')

# Plot truth position, spline mpe, and max likelihood
ax.plot(0, 0, 'rx', markersize=10, label='Truth')
ax.plot(np.degrees(spline_az - truth_az),
        np.degrees(spline_ze - truth_ze), 
        'gx', markersize=10, label='Spline MPE')
ax.plot(np.degrees(max_llh_az - truth_az), 
        np.degrees(max_llh_ze - truth_ze), 
        'bx', markersize=10, label='Max Likelihood')

# Add a contour line
clevel = -2.305
cs = ax.contour(np.degrees(azimuth), 
                np.degrees(zenith), 
                delta_log_skymap, levels=[clevel], 
                linewidths=1, linestyles='solid')


contour_line = Line2D([0], [0], color=cs.collections[0].get_edgecolor()[0], 
                      linewidth=1, linestyle='solid')
handles, labels = ax.get_legend_handles_labels()
handles.append(contour_line)
labels.append(f'90% contour')

# Add grid, labels, and legend
ax.grid(True)
ax.set_xlabel(r'$\Delta$ azimuth [deg]')
ax.set_ylabel(r'$\Delta$ zenith [deg]')
plt.legend(handles, labels)

# Finalize and show the plot
plt.tight_layout()
plt.savefig(f'{model_path}/event_llh_plot_{event_no[0]}.pdf')
plt.show()
plt.close()