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
import pathlib
import re
import os
import scipy.stats as stats
import numpy as np
import logging
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

# remove FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.switch_backend('agg')

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def load_path(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    
    return all_files

import matplotlib.pyplot as plt
import numpy as np
import os

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
    ax1.set_title('%s' % name, pad=40, fontsize=TITLE_SIZE)
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
    UniqueExperiments = data['Experiment'].unique().astype(str)
    N_exp = int(len(UniqueExperiments))

    dt = int(conf['everyXhourField'])
    N_steps = int(round((data['ElapsedTime (h)'].max()+1) / dt, 0))
    
    # Create a text file to store the results
    reportPath = os.path.join(conf['MainFolder'],'Report', 'Temporal Parameters')
    
    if "/" in metric:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric.replace('/',' over '))
    else:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric)
        
    # First row should say "Using Mann Whitney U test to compare the growth speed of different experiments"
    with open(reportPath_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        f.write('Uses the average value, per plant, per day\n\n')
         
        for step in range(0, N_steps):            
            end = dt * (step+1)
            end = int(min(end, data['ElapsedTime (h)'].max()))
            hours = np.arange(dt * step, end)
            subdata = data[data['ElapsedTime (h)'].isin(hours)]

            if conf['averagePerPlantStats']:
                subdata = subdata.groupby(['Experiment', 'Plant_id']).mean(numeric_only=True).reset_index()
    
            subdata['Experiment'] = subdata['Experiment'].astype(str)
            
            # Compare every pair of experiments with Mann-Whitney U test
            f.write('Hours from ' + str(step*dt) + ' to ' + str(end) + '\n')
            
            for i in range(0, N_exp-1):
                for j in range(i+1, N_exp):
                    exp1 = subdata[subdata['Experiment'] == UniqueExperiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == UniqueExperiments[j]][metric]
                    
                    # Perform Mann-Whitney U test
                    try:
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)
                        
                        # Write the number of samples in each experiment, both in the same line
                        f.write('Number of samples ' + UniqueExperiments[i] + ': ' + str(len(exp1)) + ' - ')
                        f.write('Number of samples ' + UniqueExperiments[j] + ': ' + str(len(exp2)) + '\n')
                        
                        # Write the mean value of each experiment
                        f.write('Mean ' + UniqueExperiments[i] + ': ' + str(round(exp1.mean(), 2)) + ' - ')
                        f.write('Mean ' + UniqueExperiments[j] + ': ' + str(round(exp2.mean(), 2)) + '\n')
                        
                        # Compare the p-value with the significance level
                        if p < 0.05:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are significantly different. P-value: ' + str(p) + '\n')
                        else:
                            f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' are not significantly different. P-value: ' + str(p) + '\n')
                    except:
                        f.write('Experiments ' + UniqueExperiments[i] + ' and ' + UniqueExperiments[j] + ' could not be compared\n')
                        
            f.write('\n')            
    return

def generateTableTemporal(conf, data):
    dt = int(conf['everyXhourField'])
    N_steps = int(round((data['ElapsedTime (h)'].max()+1) / dt, 0))
    reportPath = os.path.join(conf['MainFolder'],'Report', 'Temporal Parameters')
    
    summaryDF = []
    
    for step in range(0, N_steps):            
        end = dt * (step+1)
        end = int(min(end, data['ElapsedTime (h)'].max()))
        hours = np.arange(dt * step, end)
        subdata = data[data['ElapsedTime (h)'].isin(hours)]
        subdata = subdata.groupby(['Experiment', 'Plant_id']).mean(numeric_only=True).reset_index()
        subdata = subdata.groupby(['Experiment']).agg({'MainRootLength (mm)': ['count', 'mean', 'std'],
                                                      'LateralRootsLength (mm)': ['mean', 'std'], 
                                                      'TotalLength (mm)': ['mean', 'std'], 
                                                      'NumberOfLateralRoots': ['mean', 'std'], 
                                                      'DiscreteLateralDensity (LR/cm)': ['mean', 'std'], 
                                                      'MainOverTotal (%)': ['mean', 'std'],
                                                      'HypocotylLength (mm)': ['mean', 'std']})
        
        subdata.columns = [' '.join(col).strip() for col in subdata.columns.values]
        subdata = subdata.reset_index()
        subdata['Hours interval'] = str(dt * step) + '-' + str(end - 1)
        summaryDF.append(subdata)

    summaryDF = pd.concat(summaryDF)
    summaryDF.rename(columns={"MainRootLength (mm) count": "N experiment"}, inplace=True)
    col = summaryDF.pop("Hours interval")
    summaryDF.insert(0, col.name, col)
    summaryDF.to_csv(os.path.join(reportPath, "Temporal Parameters Summary Table.csv"), index=False)    
    
def plot_info_all(savepath, dataframe):
    plt.ioff()
    
    # set color palette
    sns.set_palette("tab10")

    # plt.rcParams.update({'font.size': 18})

    fig3 = plt.figure(figsize=(12,8), constrained_layout=True)
    gs = fig3.add_gridspec(2, 3)
    f_ax1 = fig3.add_subplot(gs[0, 0])
    f_ax2 = fig3.add_subplot(gs[0, 1])
    f_ax3 = fig3.add_subplot(gs[0, 2])
    f_ax4 = fig3.add_subplot(gs[1, 0])
    f_ax5 = fig3.add_subplot(gs[1, 1])
    f_ax6 = fig3.add_subplot(gs[1, 2])

    sns.lineplot(x = 'ElapsedTime (h)', y = 'MainRootLength (mm)', data = dataframe, hue = 'Experiment', errorbar='se', ax = f_ax1)
    f_ax1.set_title('MR length', fontsize = 16)
    f_ax1.set_ylabel('Length (mm)', fontsize = 12)
    f_ax1.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax1.legend(loc='upper left')

    sns.lineplot(x = 'ElapsedTime (h)', y = 'LateralRootsLength (mm)', data = dataframe, hue = 'Experiment', errorbar='se', ax = f_ax2)
    f_ax2.set_title('LR length', fontsize = 16)
    f_ax2.set_ylabel('Length (mm)', fontsize = 12)
    f_ax2.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax2.legend(loc='upper left')

    sns.lineplot(x = 'ElapsedTime (h)', y = 'TotalLength (mm)', data = dataframe, hue = 'Experiment', errorbar='se', ax = f_ax3)
    f_ax3.set_title('TR length', fontsize = 16)
    f_ax3.set_ylabel('Length (mm)', fontsize = 12)
    f_ax3.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax3.legend(loc='upper left')
        
    sns.lineplot(x = 'ElapsedTime (h)', y = 'NumberOfLateralRoots', hue = 'Experiment', data = dataframe, errorbar='se', ax = f_ax4)
    f_ax4.set_title('Number of LR', fontsize = 16)
    f_ax4.set_ylabel('Number of LR', fontsize = 12)
    f_ax4.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax4.legend(loc='upper left')

    sns.lineplot(x = 'ElapsedTime (h)', y = 'DiscreteLateralDensity (LR/cm)', hue = 'Experiment', data = dataframe, errorbar='se', ax = f_ax5)
    f_ax5.set_title('Discrete LR Density', fontsize = 16)
    f_ax5.set_ylabel('Discrete LR density (LRs/cm)', fontsize = 12)
    f_ax5.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax5.legend(loc='upper left')

    sns.lineplot(x = 'ElapsedTime (h)', y = 'MainOverTotal (%)', hue = 'Experiment', data = dataframe, errorbar='se', ax = f_ax6)
    f_ax6.set_title('MR length / TR length (%)', fontsize = 16)
    f_ax6.set_ylabel('Percentage (%)', fontsize = 12)
    f_ax6.set_xlabel('Elapsed Time (h)', fontsize = 12)
    f_ax6.legend(loc='lower left')
    
    """
    f_ax1.annotate('(A)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax2.annotate('(B)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax3.annotate('(C)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax4.annotate('(D)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax5.annotate('(E)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax6.annotate('(F)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    """
    
    plt.savefig(os.path.join(savepath,'Temporal_Subplots_Mean_SE.svg'),dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(savepath,'Temporal_Subplots_Mean_SE.png'),dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')
    
    # List of metrics to plot individually
    metrics = [
        ('MainRootLength (mm)', 'Main Root Length'),
        ('LateralRootsLength (mm)', 'Lateral Roots Length'),
        ('TotalLength (mm)', 'Total Length'),
        ('NumberOfLateralRoots', 'Number of Lateral Roots'),
        ('DiscreteLateralDensity (LR/cm)', 'Discrete Lateral Density'),
        ('MainOverTotal (%)', 'Main Root over Total Length'),
        ('HypocotylLength (mm)', 'Hypocotyl Length')
    ]

    for column, title in metrics:
        plt.ioff()
        plt.figure(figsize=(8, 6))
        
        sns.lineplot(
            x='ElapsedTime (h)', 
            y=column, 
            data=dataframe, 
            hue='Experiment', 
            errorbar='se'
        )
        
        plt.title(f'{title} Over Time', fontsize=16)
        plt.ylabel(column, fontsize=12)
        plt.xlabel('Elapsed Time (h)', fontsize=12)
        plt.legend(loc='best')
        
        # Create a filename-friendly string (removing special characters)
        filename = column.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'Pct')
        
        plt.savefig(os.path.join(savepath, f'{filename}_Temporal.png'), dpi=300, bbox_inches='tight')
        
        # Clean up memory after each plot
        plt.close()

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
    return