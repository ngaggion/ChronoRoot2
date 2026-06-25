""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy.stats as stats
import numpy as np
import logging
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

# remove FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.switch_backend('agg')

from .utils.fileUtilities import convertFromPathSafe, convertToPathSafe
from .utils.report_paths import (
    MODULE_TEMPORAL,
    metric_dir,
    overview_dir,
    table_file,
    temporal_metric_slug,
)
from .stats_utils import perform_temporal_pairwise_stats, ensure_factor_columns
from .report_plots import emit_temporal_comparison_plots
from .utils.report_style import genotype_palette_for_data, get_genotype_axis_label


def plot_individual_plant(savepath, dataframe, name):
    plt.ioff()
    
    # Define font sizes for consistency across subplots
    LABEL_SIZE = 18
    TICK_SIZE = 16
    TITLE_SIZE = 18
    LEGEND_SIZE = 16
    DAY_TICK_SIZE = 12

    # Create subplots: 2 rows, 1 column.
    # Increase figure height (e.g., (9, 10)) to accommodate two plots.
    # sharex=True ensures they align and only the bottom plot shows hour labels.
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 10), dpi=300, sharex=True)

    # ===========================
    # TOP SUBPLOT (Lengths)
    # ===========================
    # Plot MainRootLength and LateralRootsLength on the top axis (ax1)
    # Adding distinct labels for the legend
    dataframe.plot(x='ElapsedTime (h)', y='MainRootLength (mm)', ax=ax1, color='g', label='Main Root Length')
    dataframe.plot(x='ElapsedTime (h)', y='LateralRootsLength (mm)', ax=ax1, color='b', label='Lateral Roots Length')
    dataframe.plot(x='ElapsedTime (h)', y='HypocotylLength (mm)', ax=ax1, color='r', label='Hypocotyl Length')
    
    # Increase title padding to make room for the top "Days" axis ticks
    ax1.set_title('%s' % convertFromPathSafe(name), pad=40, fontsize=TITLE_SIZE)
    ax1.set_ylabel('Length (mm)', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='y', which='major', labelsize=TICK_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, loc='upper left')
    # Remove x-label from top plot since it's shared
    ax1.set_xlabel('')

    # ===========================
    # BOTTOM SUBPLOT (Number of LRs)
    # ===========================
    # Plot NumberOfLateralRoots on the bottom axis (ax2)
    # Using magenta ('m') for contrast
    dataframe.plot(x='ElapsedTime (h)', y='NumberOfLateralRoots', ax=ax2, color='m', legend=False)
    
    ax2.set_ylabel('Number of Lateral Roots', fontsize=LABEL_SIZE)
    ax2.set_xlabel('Elapsed Time (h)', fontsize=LABEL_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # ===========================
    # SECOND X-AXIS (DAYS) ON TOP
    # ===========================
    # Create the twin axis attached to the TOP subplot (ax1)
    ax1_days = ax1.twiny()
    
    # Ensure the limits match the shared x-axis
    ax1_days.set_xlim(ax1.get_xlim())

    # Calculate the total number of days
    max_hours = dataframe['ElapsedTime (h)'].max()
    # Handle potential empty plots or very short times
    if pd.notna(max_hours) and max_hours > 0:
        total_days = np.ceil(max_hours / 24).astype(int)

        # Create day ticks if the experiment is longer than 24h
        if total_days > 0:
            day_ticks = np.arange(24, total_days * 24 + 1, 24)
            day_labels = [f'Day {i}' for i in range(1, total_days + 1)]

            # Set day ticks and labels
            ax1_days.set_xticks(day_ticks)
            ax1_days.set_xticklabels(day_labels, rotation=45, ha='left', fontsize=DAY_TICK_SIZE)
        else:
             ax1_days.set_xticks([])
    else:
        ax1_days.set_xticks([])

    # Customize the appearance of the top ticks
    ax1_days.tick_params(axis='x', which='major', length=8, width=2, color='black')
    ax1_days.tick_params(axis='x', which='minor', length=4, width=1, color='black')
    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    fig.savefig(os.path.join(savepath, name), dpi=300, bbox_inches='tight')
        
    plt.cla()
    plt.clf()
    plt.close('all')

def performStatisticalAnalysis(conf, data, metric):
    data = ensure_factor_columns(data)
    slug = temporal_metric_slug(metric)
    table_path = table_file(conf, MODULE_TEMPORAL, slug, 'summary_table.csv')
    perform_temporal_pairwise_stats(
        conf, data, metric,
        module=MODULE_TEMPORAL, metric_slug_name=slug,
        table_file_path=table_path,
    )
    _write_metric_summary_table(conf, data, metric, slug)
    base_dir = metric_dir(conf, MODULE_TEMPORAL, slug)
    emit_temporal_comparison_plots(
        conf, data, metric, base_dir,
        module=MODULE_TEMPORAL, metric_slug_name=slug,
        metric_label=metric,
    )
    return


def _write_metric_summary_table(conf, data, metric, slug):
    """Per-metric descriptive summary across intervals and grouping factors."""
    dt = int(conf['everyXhourField'])
    max_hour = data['ElapsedTime (h)'].max()
    if pd.isna(max_hour):
        return

    n_steps = int(round((max_hour + 1) / dt, 0))
    rows = []
    for step in range(n_steps):
        end = int(min(dt * (step + 1), max_hour))
        hours = np.arange(dt * step, end)
        subdata = data[data['ElapsedTime (h)'].isin(hours)]
        subdata = subdata.groupby(
            ['Experiment', 'PlateCondition', 'ExtraVariable', 'Plant_id']
        ).mean(numeric_only=True).reset_index()
        grouped = subdata.groupby(['Experiment', 'PlateCondition', 'ExtraVariable']).agg(
            n_plants=('Plant_id', 'nunique'),
            mean=(metric, 'mean'),
            sd=(metric, 'std'),
        ).reset_index()
        grouped['hours_interval'] = f'{dt * step}-{end - 1}'
        rows.append(grouped)

    if not rows:
        return

    result = pd.concat(rows, ignore_index=True)
    result = result.round(3)
    result.to_csv(table_file(conf, MODULE_TEMPORAL, slug, 'summary_table.csv'), index=False)

def _build_temporal_summary_table(data, group_cols, dt, max_hour):
    n_steps = int(round((max_hour + 1) / dt, 0))
    summary_df = []

    agg_cols = {
        'MainRootLength (mm)': ['count', 'mean', 'std'],
        'LateralRootsLength (mm)': ['mean', 'std'],
        'TotalLength (mm)': ['mean', 'std'],
        'NumberOfLateralRoots': ['mean', 'std'],
        'DiscreteLateralDensity (LR/cm)': ['mean', 'std'],
        'MainOverTotal (%)': ['mean', 'std'],
        'HypocotylLength (mm)': ['mean', 'std'],
    }

    for step in range(n_steps):
        end = int(min(dt * (step + 1), max_hour))
        hours = np.arange(dt * step, end)
        subdata = data[data['ElapsedTime (h)'].isin(hours)]
        subdata = subdata.groupby(group_cols + ['Plant_id']).mean(numeric_only=True).reset_index()
        subdata = subdata.groupby(group_cols).agg(agg_cols)
        subdata.columns = [' '.join(col).strip() for col in subdata.columns.values]
        subdata = subdata.reset_index()
        subdata['Hours interval'] = f'{dt * step}-{end - 1}'
        summary_df.append(subdata)

    if not summary_df:
        return pd.DataFrame()

    result = pd.concat(summary_df)
    if 'MainRootLength (mm) count' in result.columns:
        result.rename(columns={'MainRootLength (mm) count': 'N experiment'}, inplace=True)
    col = result.pop('Hours interval')
    result.insert(0, col.name, col)
    return result


def generateTableTemporal(conf, data):
    data = ensure_factor_columns(data)
    dt = int(conf['everyXhourField'])
    max_hour = data['ElapsedTime (h)'].max()

    tables = [
        (_build_temporal_summary_table(data, ['Experiment'], dt, max_hour), 'summary_by_genotype.csv'),
        (_build_temporal_summary_table(data, ['PlateCondition', 'Experiment'], dt, max_hour),
         'summary_by_plate.csv'),
        (_build_temporal_summary_table(data, ['ExtraVariable', 'Experiment'], dt, max_hour),
         'summary_by_extra_variable.csv'),
    ]

    for table, filename in tables:
        if not table.empty:
            table.to_csv(os.path.join(overview_dir(conf, MODULE_TEMPORAL), filename), index=False)
    
def plot_info_all(conf, dataframe):
    plt.ioff()
    dataframe = ensure_factor_columns(dataframe)
    geno_palette = genotype_palette_for_data(dataframe)
    geno_label = get_genotype_axis_label(conf)

    def _plot_metric(ax, y_col, title):
        sns.lineplot(
            x='ElapsedTime (h)', y=y_col, data=dataframe, hue='Experiment',
            errorbar='se', ax=ax, palette=geno_palette,
        )
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', title=geno_label)

    fig3 = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig3.add_gridspec(2, 3)
    axes = [fig3.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    metrics = [
        ('MainRootLength (mm)', 'Main root length'),
        ('LateralRootsLength (mm)', 'Lateral root length'),
        ('TotalLength (mm)', 'Total root length'),
        ('NumberOfLateralRoots', 'Number of lateral roots'),
        ('DiscreteLateralDensity (LR/cm)', 'Discrete lateral root density'),
        ('MainOverTotal (%)', 'Main root / total length (%)'),
    ]
    for ax, (col, title) in zip(axes, metrics):
        _plot_metric(ax, col, title)
        ax.set_xlabel('Elapsed Time (h)', fontsize=12)
        if col == 'NumberOfLateralRoots':
            ax.set_ylabel('Number of LR', fontsize=12)
        elif col == 'DiscreteLateralDensity (LR/cm)':
            ax.set_ylabel('Discrete LR density (LRs/cm)', fontsize=12)
        elif col == 'MainOverTotal (%)':
            ax.set_ylabel('Percentage (%)', fontsize=12)
        else:
            ax.set_ylabel('Length (mm)', fontsize=12)

    plt.savefig(os.path.join(overview_dir(conf, MODULE_TEMPORAL), 'all_metrics_subplots.png'), dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')