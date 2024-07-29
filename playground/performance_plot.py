import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = './plots_07_23_1'

df = pd.read_pickle(f'{path}/performance.pkl')

def angle(az_true, zen_true, az_pred, zen_pred):

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

df['max-truth'] = angle(df['truth_az'], df['truth_ze'],df['max_llh_az'], df['max_llh_ze'])
df['max-spline'] = angle(df['spline_az'], df['spline_ze'],df['max_llh_az'], df['max_llh_ze'])
df['spline-truth'] = angle(df['truth_az'], df['truth_ze'],df['spline_az'], df['spline_ze'])


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

print(np.mean(df['max-truth']))
plot_percentiles(df, 'max-truth', 'Max LLH to Truth', f'{path}/max_llh_to_truth.pdf')
plot_percentiles(df, 'max-spline', 'Max LLH to Spline', f'{path}/max_llh_to_spline.pdf')
plot_percentiles(df, 'spline-truth', 'Spline to Truth', f'{path}/spline_to_truth.pdf')
