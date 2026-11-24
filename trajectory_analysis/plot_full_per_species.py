#%% Import packages
import os
os.chdir(r"C:\Users\IsaH\Documents\Github\Point-light-source\Tracking")

import os
import numpy as np
import pandas as pd
import matplotlib as mpl

# Set font for all plots
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10          # Optional default font size
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8

# Save text in SVGs as real text (not paths), so it can be edited in Inkscape
mpl.rcParams['svg.fonttype'] = 'none'

# Math text ($...$) in Arial, the same font as the rest of the figure (italic variables, upright numbers)
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from post_processing_functions import *
import pickle
from scipy.stats import gaussian_kde
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy as sp
from scipy import stats


# Create a single A4-sized Figure and a 1x9 grid with small spacing
mm_to_inch = lambda mm: mm / 25.4
A4_W_MM = 210.0
fig_w = mm_to_inch(18)
fig_h = mm_to_inch(18*1.5)

# Polar histogram of the swimming direction, optionally weighted by speed (WEIGHTED_POLAR)
def polar_hist(theta, speed, edges):
    weights = speed if WEIGHTED_POLAR else None
    counts, _ = np.histogram(theta, bins=edges, weights=weights, density=True)
    return counts

# Shared radial scale for the polar histograms of one figure: 10% above the highest bar/line,
# so nothing leaves the plot and the panels of one figure stay comparable
def set_polar_scale(axes, peak):
    ylim_pol = 1.1 * peak
    for ax in axes:
        ax.set_ylim(0, ylim_pol)
        ax.set_yticks([ylim_pol/3, 2*ylim_pol/3])
        ax.set_yticklabels(['', ''])

# Density of the vertical velocity dy (vertical histogram), optionally weighted by speed (WEIGHTED_VERT)
def vert_kde(d):
    d = d.dropna(subset=['dy', 'speed'] if WEIGHTED_VERT else ['dy'])
    weights = d['speed'].values if WEIGHTED_VERT else None
    dy = d['dy'].values
    # KDE_BW in mm/s; gaussian_kde expects a factor relative to the standard deviation of the data
    bw = None if KDE_BW is None else KDE_BW / np.std(dy)
    return gaussian_kde(dy, weights=weights, bw_method=bw)

# Rolling mean of dy over `window` frames within each track (tank + particle), stored in the column 'smooth_dy'
def add_smooth_dy(d):
    d = d.sort_values(['tank', 'particle', 't']).reset_index(drop=True)
    d['smooth_dy'] = d.groupby(['tank', 'particle'])['dy'].transform(
        lambda x: x.rolling(window, min_periods=1, center=True).mean())
    return d

# Largest absolute boxplot whisker (1.5 x IQR rule, same as matplotlib's boxplot) of one panel's dy values
def whisker_extent(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low = values[values >= q1 - 1.5*iqr].min()
    high = values[values <= q3 + 1.5*iqr].max()
    return max(abs(low), abs(high))

# Symmetric vertical axis range: the manual value, raised (rounded up to 0.1 mm/s) if a boxplot whisker
# of any panel plus YLIM_MARGIN would not fit
def viol_ylim(lim_manual, panels):
    lim_whisker = max(whisker_extent(v) for v in panels) * (1 + YLIM_MARGIN)
    if lim_whisker > lim_manual:
        lim = np.ceil(lim_whisker * 10) / 10
        print(f"ylim {lim_manual} mm/s is smaller than the whiskers, raised to {lim} mm/s")
        return lim
    return lim_manual

# Ticks for the symmetric vertical axis (0 in the centre), every 0.5 (below 2), 1 (up to 5) or 2 mm/s;
# works for non-integer ranges (ticks at the multiples of the step that fit inside the range)
def viol_ticks(lim):
    step = 0.5 if lim < 2 else 1 if lim <= 5 else 2
    n = int(np.floor(lim / step + 1e-9))
    return list(np.arange(-n, n + 1) * step)

# Density of dy on y_grid with the area normalized to 1 (TEST: estimated from 100 random frames only)
def vert_density(d, y_grid):
    if TEST:
        d = d.sample(n=min(100, len(d)))
    density = vert_kde(d)(y_grid)
    return density / np.trapezoid(density, y_grid)

# Circular variance (1 - mean resultant length), 0 = all in the same direction, 1 = uniformly spread
def circular_variance(theta, weights=None):
    if weights is None:
        weights = np.ones_like(theta)
    R = np.abs(np.sum(weights * np.exp(1j * theta))) / np.sum(weights)
    return 1 - R

# Load path
path_to_tracking_data = r"\\lces1133cs.uw.lu.se\LU24A1047-PLS$\TrackingData" #Research data folder

#%% Prepare data for plot
TEST = False # Set to True for a quick test run: densities from 100 random frames only and figures are not saved
per_traj = False # Set to True to use per-trajectory data, False for per-frame data
WEIGHTED_POLAR = True # Set to True to weight the polar histograms by speed (direction of the distance travelled)
WEIGHTED_VERT = False # Set to True to weight the vertical histograms by speed (counts the vertical speed twice)
SMOOTH_DY = True # Set to True to smooth the dy values with a rolling mean (window=5 frames) before plotting
KDE_BW = 0.05 # Width [mm/s] of the Gaussian smoothing of the dy density curves; smaller = more detail, None = automatic (Scott's rule)
window = 20
YLIM_MARGIN = 0.1 # If the manual ylim is smaller than the largest boxplot whisker, ylim is raised to the whisker + this fraction

# Define measurement 
# measurement = 'Cladocerans_2605'
measurement = 'Artemia_0805'
# measurement = 'BarnacleLarvae_2102'  

### Load data
path_to_trajectories = os.path.join(path_to_tracking_data, measurement, 'trajectories_scaled')

df_still = pickle.load(open(os.path.join(path_to_trajectories, 'traj_still_scaled.pkl'), 'rb'))
df_breeze = pickle.load(open(os.path.join(path_to_trajectories, 'traj_breeze_scaled.pkl'), 'rb'))
df_stormy = pickle.load(open(os.path.join(path_to_trajectories, 'traj_stormy_scaled.pkl'), 'rb'))
df_nothing = pickle.load(open(os.path.join(path_to_trajectories, 'traj_nothing_scaled.pkl'), 'rb'))

df_still = add_smooth_dy(df_still)
df_breeze = add_smooth_dy(df_breeze)
df_stormy = add_smooth_dy(df_stormy)
df_nothing = add_smooth_dy(df_nothing)

path_to_figure_savelocation = os.path.join(path_to_tracking_data, "manus_figures", 'verthist_polarhist_perframe_combined')
path_to_figure_savelocation_overlay = os.path.join(path_to_tracking_data, "manus_figures", 'verthist_polarhist_perframe_overlay')
save_control_main = os.path.join(path_to_tracking_data, "manus_figures", "control_species_combined")
save_control_overlay = os.path.join(path_to_tracking_data, "manus_figures", "control_species_overlay")

os.makedirs(save_control_main, exist_ok=True)
os.makedirs(save_control_overlay, exist_ok=True)
os.makedirs(path_to_figure_savelocation, exist_ok=True)
os.makedirs(path_to_figure_savelocation_overlay, exist_ok=True)
### Calculate mean and standard error for each treatment
# Define treatments
treatments = ['still', 'breeze', 'stormy', 'control']
conditions = ['still', 'breeze', 'stormy']

if measurement == 'Cladocerans_2605':
    tanks = [1, 2, 3, 4, 5]
else:
    tanks = [1, 2, 3]

# Extract data for each treatment
still_means = [np.mean(df_still['dy'][df_still['tank'] == tank]) for tank in tanks]
breeze_means = [np.mean(df_breeze['dy'][df_breeze['tank'] == tank]) for tank in tanks]
stormy_means = [np.mean(df_stormy['dy'][df_stormy['tank'] == tank]) for tank in tanks]
control_means = [np.mean(df_nothing['dy'][df_nothing['tank'] == tank]) for tank in tanks]

### Calculate mean and standard error for each treatment
means = [np.mean(still_means), np.mean(breeze_means), np.mean(stormy_means), np.mean(control_means)]
STDs = [np.std(still_means), np.std(breeze_means), np.std(stormy_means), np.std(control_means)]
 
# create dataframe for plotting
if SMOOTH_DY:
    data = {'dy': np.concatenate([df_still['smooth_dy'].values, df_breeze['smooth_dy'].values, df_stormy['smooth_dy'].values]),
            'speed': np.concatenate([df_still['speed'].values, df_breeze['speed'].values, df_stormy['speed'].values]),
            'orientation': np.concatenate([df_still['orientation'].values, df_breeze['orientation'].values, df_stormy['orientation'].values]),
            'tank': np.concatenate([df_still['tank'].values, df_breeze['tank'].values, df_stormy['tank'].values]),
            'Condition': ['still'] * len(df_still) + ['breeze'] * len(df_breeze) + ['stormy'] * len(df_stormy)}
else:
    data = {'dy': np.concatenate([df_still['dy'].values, df_breeze['dy'].values, df_stormy['dy'].values]),
        'speed': np.concatenate([df_still['speed'].values, df_breeze['speed'].values, df_stormy['speed'].values]),
        'orientation': np.concatenate([df_still['orientation'].values, df_breeze['orientation'].values, df_stormy['orientation'].values]),
        'tank': np.concatenate([df_still['tank'].values, df_breeze['tank'].values, df_stormy['tank'].values]),
        'Condition': ['still'] * len(df_still) + ['breeze'] * len(df_breeze) + ['stormy'] * len(df_stormy)}

df = pd.DataFrame(data)
numobs = []
# Before normalization
for condition in conditions:
    # print(f"\n{condition}")
    for tank in tanks:
        n = len(df[(df['Condition'] == condition) & (df['tank'] == tank)])
        # print(f"  tank {tank}: {n:,} observations")
        numobs.append(n)

# normalize number of counts per tank to the same number for each condition to avoid bias in the density estimation and plotting
min_numobs = min(numobs)

for condition in conditions:
    for tank in tanks:
        df_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)]
        if len(df_tank) > min_numobs:
            df_tank_sampled = df_tank.sample(n=min_numobs, random_state=42)
            df = df.drop(df_tank.index)
            df = pd.concat([df, df_tank_sampled], ignore_index=True)

xlim_viol = 0.25  # horizontal axis range [mm/s] for the violin plots, None = automatic

if measurement == 'BarnacleLarvae_2102':
    ylim_viol_manual = 2  # vertical axis range [mm/s], raised automatically if the whiskers do not fit

elif measurement == 'Artemia_0805':
    ylim_viol_manual = 6  # vertical axis range [mm/s], raised automatically if the whiskers do not fit

elif measurement == 'Cladocerans_2605':
    ylim_viol_manual = 3.5  # vertical axis range [mm/s], raised automatically if the whiskers do not fit

else:
    raise ValueError("Measurement not recognized. Please check the measurement variable.")

# Symmetric vertical axis range: manual value per measurement, raised if needed to fit the boxplot whiskers
ylim_viol_main = viol_ylim(ylim_viol_manual, [df[df['Condition'] == c]['dy'].dropna().values for c in conditions])
ytick_list = viol_ticks(ylim_viol_main)
print(f"Vertical axis range: ±{ylim_viol_main} mm/s")

### Statistical analysis

# Calculate mean dy and weighted circular variance of the swimming direction for each replicate
# and save results to a dataframe for further analysis
results = []

for condition in conditions:
    dy_tank = []
    vw_tank = []
    for tank in tanks:
        d = df[(df['Condition'] == condition) & (df['tank'] == tank)].dropna(subset=['dy', 'orientation', 'speed'])
        theta = np.mod(d['orientation'].values, 2*np.pi)

        dy_mean = np.mean(d['dy'])
        dy_tank.append(dy_mean)

        vw = circular_variance(theta, d['speed'].values)
        vw_mean = np.mean(vw)
        vw_tank.append(vw_mean)

        results.append({
            'Condition': condition,
            'Tank': tank,
            'dy_mean': dy_mean,
            'vw_mean': vw_mean,
        })

    # Mean ± standard error over the tanks (SE = SD / sqrt(number of tanks))
    dy_mean_tank = np.mean(dy_tank)
    vw_mean_tank = np.mean(vw_tank)
    dy_se = np.std(dy_tank, ddof=1) / np.sqrt(len(tanks))
    vw_se = np.std(vw_tank, ddof=1) / np.sqrt(len(tanks))

    print(f"  {condition}: mean dy = {dy_mean_tank:.3f} ± {dy_se:.3f} (SE), circular variance (speed weighted) = {vw_mean_tank:.3f} ± {vw_se:.3f} (SE), n = {len(tanks)} tanks")

### Run anova on dy and circular variance across conditions, using the tank means as replicates
df_results = pd.DataFrame(results)
alpha = 0.05
tukey_letters = {}   # tukey_letters[variable][condition] = letters, used in the plots

for variable in ['dy_mean', 'vw_mean']:
    samples = [df_results[df_results['Condition'] == c][variable].values for c in conditions]
    f, p = stats.f_oneway(*samples)
    print(f"\n{variable}: ANOVA F = {f:.2f}, p = {p:.4f}")

    ### Run Tukey's HSD post-hoc test for pairwise comparisons across conditions
    p_tukey = stats.tukey_hsd(*samples).pvalue
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            print(f"  {conditions[i]} vs {conditions[j]}: p = {p_tukey[i, j]:.4f}")

    # Compact letter display: treatments that share a letter are not significantly different
    order = list(np.argsort([-np.mean(s) for s in samples]))   # highest mean first
    letter_sets = []
    for i in order:
        group = []
        for j in order:
            if p_tukey[i, j] >= alpha and all(p_tukey[j, k] >= alpha for k in group):
                group.append(j)
        if not any(set(group) <= s for s in letter_sets):
            letter_sets.append(set(group))
    tukey_letters[variable] = {}
    for i in range(len(conditions)):
        letters = ''.join('abcdefgh'[n] for n, s in enumerate(letter_sets) if i in s)
        tukey_letters[variable][conditions[i]] = letters
        print(f"  {conditions[i]:>8}: {letters}")


### Make the plot
conditions = ['still', 'breeze', 'stormy']
colors = sns.color_palette("Blues", 4)[1:]
# colors = sns.color_palette("Set2", 3)

# Create figure with cloud and boxplot for each condition in separate subplots, and polar histogram with swimming angle distribution on top of each subplot # Add some extra space for the cloud and boxplot
cloud_width = xlim_viol * 0.7  # Max width of the density cloud (80% of xlim)
viol_curve_frac = 0.75  # Peak of the widest density curve in a figure, as fraction of xlim (scaled automatically)
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4  # Boxplot width as a fraction of cloud width
box_linewidth = 1

w_main_mm = 210/3*0.9
f_main_height = 1.3

max_points = len(data) # subsample per treatment for plotting only
points_scatter = 3000

cloud_x = cloud_width - cloud_offset

fig = plt.figure(figsize=(mm_to_inch(w_main_mm), mm_to_inch(w_main_mm) * f_main_height))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection='polar') for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

peak_pol = 0
for ax, condition, color in zip(axes_pol, conditions, colors):
    # Define treatment

    d = df[df['Condition'] == condition].dropna(subset=['orientation', 'speed'])
    orientation_vals = d['orientation'].values
    speed_vals = d['speed'].values

    # Robust, seam-free polar histogram
    theta = np.mod(orientation_vals, 2*np.pi)
    nbins = 50
    edges = np.linspace(0, 2*np.pi, nbins + 1)  # nbins + 1 edges for nbins bins
    counts = polar_hist(theta, d["speed"].values, edges)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align='edge', color=color, alpha=1, edgecolor='none')
    peak_pol = max(peak_pol, counts.max())

    ax.set_thetamin(0)

    # Speed weighted circular variance above the polar histogram: mean ± SE over the tanks, as in the statistics
    vw_values = df_results[df_results['Condition'] == condition]['vw_mean'].values
    vw_condition = np.mean(vw_values)
    vw_se = np.std(vw_values, ddof=1) / np.sqrt(len(vw_values))
    ax.set_title(rf'$V_w = {vw_condition:.2f} \pm {vw_se:.2f}$', fontsize=6, pad=4)

    # disable polar labels and ticks
    ax.set_xticklabels([])
    # ax.set_xticklabels([0, '' ,90, '', 180, '', 270, ''])

set_polar_scale(axes_pol, peak_pol)

# Density curves (area 1), scaled with one factor for the whole figure so the highest peak fits the panel
densities = {c: vert_density(df[df['Condition'] == c], y_grid) for c in conditions}
curve_width = viol_curve_frac * xlim_viol / max(dens.max() for dens in densities.values())

for ax, condition, color in zip(axes_viol, conditions, colors):

    if condition != 'still':
        ax.tick_params(axis='y', labelleft=False)
        ax.set_yticklabels([])
                
    data = df[df['Condition'] == condition]['dy'].dropna().values
    ### Density cloud
    density = densities[condition] * curve_width

    ### Plot density curves and filled area
    ax.plot(density, y_grid, color='black', linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color=color, alpha=0.7, zorder=2)
    x_jitter = -np.random.uniform(cloud_offset, cloud_width-cloud_offset, size=points_scatter)
    ax.scatter(x_jitter, np.random.choice(data, size=points_scatter, replace=False), s=1.5, color='grey', alpha=0.12, linewidth=0, zorder=4)
    # ax.errorbar(-cloud_width/2, mean, yerr=std, fmt='o', color='black', markersize=3, zorder=5)
    
    # ### Boxplot
    bp = ax.boxplot(data,
                orientation='vertical',
                positions=[-cloud_width/2],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=box_linewidth, alpha=0.8),
                medianprops=dict(color='black', linewidth=box_linewidth),
                whiskerprops=dict(color='black', linewidth=box_linewidth),
                capprops=dict(color='black', linewidth=box_linewidth),
                zorder = 5)

    ### Tukey letters (mean dy per tank), capitalized, in the top of the panel just right of the zero line
    ax.text(0.56, 0.97, tukey_letters['dy_mean'][condition].upper(), transform=ax.transAxes,
            ha='left', va='top', fontsize=8, zorder=6)

    ### Styling
    # if condition == 'breeze':
    #     ax.set_title('Distribution of dy', fontsize=10, pad=5)
    ax.set_xlabel(condition, fontsize=10)

    ax.tick_params(axis='y', labelsize=6, length=0, rotation=0)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis='y', zorder=0)
    ax.axvline(x=0, color='gray', alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main, ylim_viol_main])

    for spine in ax.spines.values():
        spine.set_visible(False)


# Add outer rectangle, same as your current plot
ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor='black',
    linewidth=1,
    zorder=100
)

ax_bg = fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_full_per_species.png'), dpi=1200, transparent=True, bbox_inches='tight')
    fig.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_full_per_species.svg'), dpi=1200, transparent=True,  bbox_inches='tight')


# Make the plot with individual overlay for each tank, to show variability across tanks, and mean ± SD for each condition

conditions = ['still', 'breeze', 'stormy']
# colors = sns.color_palette("Blues", 4)[1:]
colors = ["#0072B2","#B20000", "#E69F00", "#009E73", "#CC79A7"][:len(tanks)]

# Create figure with cloud and boxplot for each condition in separate subplots, and polar histogram with swimming angle distribution on top of each subplot # Add some extra space for the cloud and boxplot
cloud_width = xlim_viol * 0.7  # Max width of the density cloud (80% of xlim)
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4  # Boxplot width as a fraction of cloud width
box_linewidth = 1

max_points = len(data) # subsample per treatment for plotting only
points_scatter = 3000

cloud_x = cloud_width - cloud_offset

fig = plt.figure(figsize=(mm_to_inch(w_main_mm), mm_to_inch(w_main_mm) * f_main_height))
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection='polar') for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

peak_pol = 0
for ax, condition in zip(axes_pol, conditions):
    # Define treatment

    d = df[df['Condition'] == condition].dropna(subset=['orientation', 'speed'])
    orientation_vals = d['orientation'].values
    speed_vals = d['speed'].values

    # Robust, seam-free polar histogram
    theta = np.mod(orientation_vals, 2*np.pi)
    nbins = 50
    edges = np.linspace(0, 2*np.pi, nbins + 1)  # nbins + 1 edges for nbins bins
    counts = polar_hist(theta, d["speed"].values, edges)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align='edge', color='gray', alpha=0.5, edgecolor='none')
    peak_pol = max(peak_pol, counts.max())
    
    for tank in tanks:
        d_tank = d[d['tank'] == tank]
        color = colors[tank-1]  # Assuming tank numbers start at 1 and are consecutive
        orientation_vals_tank = d_tank['orientation'].values
        theta_tank = np.mod(orientation_vals_tank, 2*np.pi)
        counts_tank = polar_hist(theta_tank, d_tank["speed"].values, edges)
        ax.plot(edges[:-1] + width/2, counts_tank, color=color, linewidth=1.2, alpha=0.8)
        peak_pol = max(peak_pol, counts_tank.max())

    ax.set_thetamin(0)

    # Speed weighted circular variance above the polar histogram: mean ± SE over the tanks, as in the statistics
    vw_values = df_results[df_results['Condition'] == condition]['vw_mean'].values
    vw_condition = np.mean(vw_values)
    vw_se = np.std(vw_values, ddof=1) / np.sqrt(len(vw_values))
    ax.set_title(rf'$V_w = {vw_condition:.2f} \pm {vw_se:.2f}$', fontsize=6, pad=4)

    # disable polar labels and ticks
    ax.set_xticklabels([])
    # ax.set_xticklabels([0, '' ,90, '', 180, '', 270, ''])

set_polar_scale(axes_pol, peak_pol)

# Density curves (area 1) of each condition and each tank, scaled with one factor for the whole figure
densities = {c: vert_density(df[df['Condition'] == c], y_grid) for c in conditions}
densities_tank = {}
for c in conditions:
    for tank in tanks:
        d_tank = df[(df['Condition'] == c) & (df['tank'] == tank)].dropna(subset=['dy'])
        if len(d_tank) > 1:
            densities_tank[(c, tank)] = vert_density(d_tank, y_grid)
peak_viol = max(dens.max() for dens in list(densities.values()) + list(densities_tank.values()))
curve_width = viol_curve_frac * xlim_viol / peak_viol

for ax, condition in zip(axes_viol, conditions):

    if condition != 'still':
        ax.tick_params(axis='y', labelleft=False)
        ax.set_yticklabels([])
                
    data = df[df['Condition'] == condition]['dy'].dropna().values
    ### Density cloud
    density = densities[condition] * curve_width

    ### Plot density curves and filled area
    # ax.plot(density, y_grid, color='black', linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color='gray', alpha=0.7, zorder=2)
    
    ### Plot overlay density curves for each tank
    for tank in tanks:
        d_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)].dropna(subset=['dy'])

        if (condition, tank) in densities_tank:
            density_tank = densities_tank[(condition, tank)] * curve_width
            color = colors[tank-1]  # Assuming tank numbers start at 1 and are consecutive
            ax.plot(density_tank, y_grid, color=color, linewidth=1, zorder=3, alpha=0.8)
            
    x_jitter = -np.random.uniform(cloud_offset, cloud_width-cloud_offset, size=points_scatter)
    ax.scatter(x_jitter, np.random.choice(data, size=points_scatter, replace=False), s=1.5, color='grey', alpha=0.12, linewidth=0, zorder=4)

    ### Boxplot
    # Tank-level mean dy points

    x_points = np.linspace(-cloud_width*0.85, -cloud_width*0.35, len(tanks))
    for x, tank in zip(x_points, tanks):
        d_tank = df[(df['Condition'] == condition) & (df['tank'] == tank)]
        tank_mean = np.mean(d_tank['dy'].dropna().values)
        tank_sd = np.std(d_tank['dy'].dropna().values, ddof=1)

        ax.scatter(x, tank_mean, s=10, color=colors[tank-1], edgecolor='none', linewidth=0.4, zorder=6)
        ax.errorbar(x, tank_mean, yerr=tank_sd, fmt='o', color=colors[tank-1], markersize=3, capsize=2, linewidth=1, zorder=7)



    # ax.errorbar(-cloud_width*0.5, mean_tank,
    #             yerr=sd_tank,
    #             fmt='o',
    #             color=colors[tank-1],
    #             markersize=3,
    #             capsize=2,
    #             linewidth=1,
    #             zorder=7)

    ### Mean ± SD
    # mean_val = np.mean(data)
    # std_val = np.std(data)

    ### Styling
    # if condition == 'breeze':
    #     ax.set_title('Distribution of dy', fontsize=10, pad=5)
    ax.set_xlabel(condition, fontsize=10)

    ax.tick_params(axis='y', labelsize=6, length=0, rotation=0)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis='y', zorder=0)
    ax.axvline(x=0, color='gray', alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main, ylim_viol_main])

    for spine in ax.spines.values():
        spine.set_visible(False)


# Add outer rectangle, same as your current plot
ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor='black',
    linewidth=1,
    zorder=100
)

ax_bg = fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(path_to_figure_savelocation_overlay, f'{measurement}_full_per_species.png'), dpi=1200, transparent=True, bbox_inches='tight')
    fig.savefig(os.path.join(path_to_figure_savelocation_overlay, f'{measurement}_full_per_species.svg'), dpi=1200, transparent=True, bbox_inches='tight')

#%% Create a flat legend for the tanks
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Tank {tank}') for i, tank in enumerate(tanks)]

fig_legend = plt.figure(figsize=(2, 0.1))
legend_ax = fig_legend.add_subplot(111)
legend_ax.legend(handles=legend_elements, ncol = len(tanks), fontsize=8)
legend_ax.axis('off')
fig_legend.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.png'), dpi=1200, transparent=True, bbox_inches='tight')
fig_legend.savefig(os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.svg'), dpi=1200, transparent=True, bbox_inches='tight')

print("saved legend at", os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.png'))
print("saved legend at", os.path.join(path_to_figure_savelocation, f'{measurement}_tank_legend.svg'))

#%% Plot control measurements for all species: no overlays + tank overlays

w_control_mm = 210/2*0.9
factor_control_height = 1.2

control_measurements = [
    "Artemia_0805",
    "Cladocerans_2605",
    "BarnacleLarvae_2102",
]

species_labels = {
    "BarnacleLarvae_2102": "Barnacle larvae",
    "Artemia_0805": "Artemia larvae",
    "Cladocerans_2605": "Water fleas",
}

control_plot_settings = {

    "ylim_viol_manual": 3.5,  # vertical axis range [mm/s], raised automatically if the whiskers do not fit
    "xlim_viol": 0.25,
    "points_scatter": 3000,
}
# Load control data
dfs_control = []

for measurement_i in control_measurements:
    path_to_trajectories_i = os.path.join(path_to_tracking_data, measurement_i, "trajectories_scaled")
    df_i = pickle.load(open(os.path.join(path_to_trajectories_i, "traj_nothing_scaled.pkl"), "rb"))

    df_i = add_smooth_dy(df_i)   # same rolling mean as the treatments
    df_i["Species"] = species_labels[measurement_i]
    df_i["Measurement"] = measurement_i

    dfs_control.append(df_i)

df_control_all = pd.concat(dfs_control, ignore_index=True)

# Normalize number of observations per species/tank
numobs = []
for measurement_i in control_measurements:
    tanks_i = [1, 2, 3, 4, 5] if measurement_i == "Cladocerans_2605" else [1, 2, 3]
    species_i = species_labels[measurement_i]

    for tank in tanks_i:
        n = len(df_control_all[(df_control_all["Species"] == species_i) & (df_control_all["tank"] == tank)])
        if n > 0:
            numobs.append(n)

min_numobs = min(numobs)

df_control_norm = []
for measurement_i in control_measurements:
    tanks_i = [1, 2, 3, 4, 5] if measurement_i == "Cladocerans_2605" else [1, 2, 3]
    species_i = species_labels[measurement_i]

    for tank in tanks_i:
        df_tank = df_control_all[(df_control_all["Species"] == species_i) & (df_control_all["tank"] == tank)]
        if len(df_tank) > min_numobs:
            df_tank = df_tank.sample(n=min_numobs, random_state=42)
        df_control_norm.append(df_tank)

df_control_plot = pd.concat(df_control_norm, ignore_index=True)

# Plot the smoothed dy, as for the treatments (SMOOTH_DY); df_control_all keeps the raw dy for the printed summaries
if SMOOTH_DY:
    df_control_plot["dy"] = df_control_plot["smooth_dy"]

### Print the circular variance of the swimming direction per species for the control (all data)
print("\nControl, circular variance of the swimming direction")
for measurement_i in control_measurements:
    species = species_labels[measurement_i]
    d = df_control_all[df_control_all["Species"] == species].dropna(subset=["orientation", "speed"])
    theta = np.mod(d["orientation"].values, 2*np.pi)
    V_tanks = [circular_variance(np.mod(d_t["orientation"].values, 2*np.pi), d_t["speed"].values)
               for _, d_t in d.groupby("tank")]
    print(f"{species:>16}: {circular_variance(theta, d['speed'].values):.3f} (speed weighted), "
          f"{circular_variance(theta):.3f} (unweighted); "
          f"per tank (speed weighted): {np.mean(V_tanks):.2f} ± {np.std(V_tanks, ddof=1):.2f} (mean ± SD, n = {len(V_tanks)})")


# Symmetric vertical axis range: manual value, raised if needed to fit the boxplot whiskers of all species
species_order = [species_labels[m] for m in control_measurements]
ylim_viol_main = viol_ylim(control_plot_settings["ylim_viol_manual"],
                           [df_control_plot[df_control_plot["Species"] == s]["dy"].dropna().values for s in species_order])
ytick_list = viol_ticks(ylim_viol_main)
print(f"Control vertical axis range: ±{ylim_viol_main} mm/s")
xlim_viol = control_plot_settings["xlim_viol"]

points_scatter = control_plot_settings["points_scatter"]

blue_colors = sns.color_palette("Blues", 4)[1:]

# color control is slightly darker than stormy
color_control = sns.color_palette("Blues", 4)[2]
tank_colors = ["#0072B2", "#B20000", "#E69F00", "#009E73", "#CC79A7"]

y_grid = np.linspace(-ylim_viol_main, ylim_viol_main, 600)

cloud_width = xlim_viol * 0.7
cloud_offset = 0.1 * xlim_viol
box_width = cloud_width * 0.4
box_linewidth = 1

# plot control, no overlays

fig = plt.figure(figsize=(
    mm_to_inch(w_control_mm),
    mm_to_inch(w_control_mm) * factor_control_height
))

gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection="polar") for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

peak_pol = 0
for ax, species in zip(axes_pol, species_order):
    d = df_control_plot[df_control_plot["Species"] == species].dropna(subset=["orientation", "speed"])

    theta = np.mod(d["orientation"].values, 2 * np.pi)
    nbins = 50
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    counts = polar_hist(theta, d["speed"].values, edges)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color=color_control, alpha=1, edgecolor="none")
    peak_pol = max(peak_pol, counts.max())

    ax.set_thetamin(0)
    ax.set_xticklabels([])

set_polar_scale(axes_pol, peak_pol)

# Density curves (area 1), scaled with one factor for the whole figure so the highest peak fits the panel
densities = {s: vert_density(df_control_plot[df_control_plot["Species"] == s], y_grid) for s in species_order}
curve_width = viol_curve_frac * xlim_viol / max(dens.max() for dens in densities.values())

for ax, species in zip(axes_viol, species_order):
    if species != species_order[0]:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values

    density = densities[species] * curve_width

    ax.plot(density, y_grid, color="black", linewidth=1, zorder=3)
    ax.fill_betweenx(y_grid, 0, density, color=color_control, alpha=0.7, zorder=2)

    n_scatter = min(points_scatter, len(data))
    x_jitter = -np.random.uniform(cloud_offset, cloud_width - cloud_offset, size=n_scatter)
    ax.scatter(
        x_jitter,
        np.random.choice(data, size=n_scatter, replace=False),
        s=1.5,
        color="grey",
        alpha=0.12,
        linewidth=0,
        zorder=4,
        rasterized=True,
    )

    ax.boxplot(
        data,
        orientation='vertical',
        positions=[-cloud_width / 2],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=box_linewidth, alpha=0.8),
        medianprops=dict(color="black", linewidth=box_linewidth),
        whiskerprops=dict(color="black", linewidth=box_linewidth),
        capprops=dict(color="black", linewidth=box_linewidth),
        zorder=5,
    )

    ax.set_xlabel(species, fontsize=10)
    ax.tick_params(axis="y", labelsize=6, length=0, rotation=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis="y", zorder=0)
    ax.axvline(x=0, color="gray", alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main, ylim_viol_main])

    for spine in ax.spines.values():
        spine.set_visible(False)

ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor="black",
    linewidth=1,
    zorder=100,
)

fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(save_control_main, "control_all_species.png"), dpi=1200, bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(save_control_main, "control_all_species.svg"), dpi=1200, bbox_inches="tight", transparent=True)

# Plot control, tank overlays

fig = plt.figure(figsize=(
    mm_to_inch(w_control_mm),
    mm_to_inch(w_control_mm) * factor_control_height
))

gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 2], wspace=0, hspace=0.3)

axes_pol = [fig.add_subplot(gs[0, i], projection="polar") for i in range(3)]
axes_viol = [fig.add_subplot(gs[1, i]) for i in range(3)]

peak_pol = 0
for ax, species, measurement_i in zip(axes_pol, species_order, control_measurements):
    d = df_control_plot[df_control_plot["Species"] == species].dropna(subset=["orientation", "speed"])

    theta = np.mod(d["orientation"].values, 2 * np.pi)
    nbins = 50
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    counts = polar_hist(theta, d["speed"].values, edges)

    width = edges[1] - edges[0]
    ax.bar(edges[:-1], counts, width=width, align="edge", color="gray", alpha=0.4, edgecolor="none")
    peak_pol = max(peak_pol, counts.max())

    tanks_i = sorted(d["tank"].dropna().astype(int).unique())

    for tank in tanks_i:
        d_tank = d[d["tank"] == tank]
        color = tank_colors[tank - 1]

        theta_tank = np.mod(d_tank["orientation"].values, 2 * np.pi)
        counts_tank = polar_hist(theta_tank, d_tank["speed"].values, edges)
        ax.plot(edges[:-1] + width / 2, counts_tank, color=color, linewidth=1.2, alpha=0.8)
        peak_pol = max(peak_pol, counts_tank.max())

    
    ax.set_thetamin(0)
    ax.set_xticklabels([])

set_polar_scale(axes_pol, peak_pol)

# Density curves (area 1) of each species and each tank, scaled with one factor for the whole figure
densities = {s: vert_density(df_control_plot[df_control_plot["Species"] == s], y_grid) for s in species_order}
densities_tank = {}
for s in species_order:
    d_s = df_control_plot[df_control_plot["Species"] == s]
    for tank in sorted(d_s["tank"].dropna().astype(int).unique()):
        d_tank = d_s[d_s["tank"] == tank].dropna(subset=["dy"])
        if len(d_tank) > 1:
            densities_tank[(s, tank)] = vert_density(d_tank, y_grid)
peak_viol = max(dens.max() for dens in list(densities.values()) + list(densities_tank.values()))
curve_width = viol_curve_frac * xlim_viol / peak_viol

for ax, species in zip(axes_viol, species_order):
    if species != species_order[0]:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])

    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values

    density = densities[species] * curve_width

    ax.fill_betweenx(y_grid, 0, density, color="gray", alpha=0.4, zorder=2)

    tanks_i = sorted(df_control_plot[df_control_plot["Species"] == species]["tank"].dropna().astype(int).unique())

    for tank in tanks_i:
        d_tank = df_control_plot[
            (df_control_plot["Species"] == species) &
            (df_control_plot["tank"] == tank)
        ].dropna(subset=["dy"])

        if (species, tank) in densities_tank:
            density_tank = densities_tank[(species, tank)] * curve_width

            color = tank_colors[tank - 1]
            ax.plot(density_tank, y_grid, color=color, linewidth=1, zorder=3, alpha=0.8)

    n_scatter = min(points_scatter, len(data))
    x_jitter = -np.random.uniform(cloud_offset, cloud_width - cloud_offset, size=n_scatter)
    ax.scatter(
        x_jitter,
        np.random.choice(data, size=n_scatter, replace=False),
        s=1.5,
        color="grey",
        alpha=0.12,
        linewidth=0,
        zorder=4,
        rasterized=True,
    )

    x_points = np.linspace(-cloud_width * 0.85, -cloud_width * 0.35, len(tanks_i))

    for x, tank in zip(x_points, tanks_i):
        d_tank = df_control_plot[
            (df_control_plot["Species"] == species) &
            (df_control_plot["tank"] == tank)
        ]

        tank_mean = np.mean(d_tank["dy"].dropna().values)
        tank_sd = np.std(d_tank["dy"].dropna().values, ddof=1)

        color = tank_colors[tank - 1]
        ax.errorbar(
            x,
            tank_mean,
            yerr=tank_sd,
            fmt="o",
            color=color,
            markersize=3,
            capsize=2,
            linewidth=1,
            zorder=7,
        )

    ax.set_xlabel(species, fontsize=10)
    ax.tick_params(axis="y", labelsize=6, length=0, rotation=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(pad=2)

    ax.grid(True, alpha=1, axis="y", zorder=0)
    ax.axvline(x=0, color="gray", alpha=1, linewidth=1, zorder=0)

    ax.set_xticks([])
    ax.set_yticks(ytick_list)
    ax.set_xlim(-xlim_viol, xlim_viol)
    ax.set_ylim([-ylim_viol_main, ylim_viol_main])

    for spine in ax.spines.values():
        spine.set_visible(False)

ax_left = axes_viol[0].get_position().x0
ax_bottom = axes_viol[0].get_position().y0
ax_right = axes_viol[-1].get_position().x1
ax_top = axes_viol[0].get_position().y1

rect = Rectangle(
    (ax_left, ax_bottom),
    ax_right - ax_left,
    ax_top - ax_bottom,
    transform=fig.transFigure,
    fill=False,
    edgecolor="black",
    linewidth=1,
    zorder=100,
)

fig.add_artist(rect)

if not TEST:
    fig.savefig(os.path.join(save_control_overlay, "control_all_species_overlay.png"), dpi=1200, bbox_inches="tight", transparent=True)
    fig.savefig(os.path.join(save_control_overlay, "control_all_species_overlay.svg"), dpi=1200, bbox_inches="tight", transparent=True)

print("Control means ± SD:")
for species in species_order:
    data = df_control_plot[df_control_plot["Species"] == species]["dy"].dropna().values
    print(f"{species}: mean = {np.mean(data):.2f}, std = {np.std(data):.2f}, n = {len(data):,} frames")
    # Per replicate: mean dy per tank (all data), then mean ± SD over the tanks
    tank_means = df_control_all[df_control_all["Species"] == species].groupby("tank")["dy"].mean()
    print(f"{'':>{len(species)}}  per tank: mean = {tank_means.mean():.2f}, SD = {tank_means.std(ddof=1):.2f}, n = {len(tank_means)} tanks")

# %%
