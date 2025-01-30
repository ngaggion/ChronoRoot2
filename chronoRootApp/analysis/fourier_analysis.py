from .report import load_path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import signal

import scipy.stats as stats

def performStatisticalAnalysis(conf, all_frames, signal = "MR"):
    UniqueExperiments = all_frames['Type'].unique()
    N_exp = int(len(UniqueExperiments))
    
    dt = int(conf['everyXhourFieldFourier'])
    N_steps = int(round((all_frames['Time'].max()+1) / dt, 0))
        
    # Create a text file to store the results
    reportPath = os.path.join(conf['MainFolder'],'Report')
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    reportPath_stats = os.path.join(reportPath_fourier, '%s_Stats.txt' % signal)

    # First row should say "Using Mann Whitney U test to compare the growth speed of different experiments"
    with open(reportPath_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare the growth speed of different experiments\n')
            
        for step in range(0, N_steps):            
            end = dt * (step+1)
            end = int(min(end, all_frames['Time'].max()))
            hours = np.arange(dt * step, end)
            subdata = all_frames[all_frames['Time'].isin(hours)]
            
            # Take average Signal per plant per experiment
            if conf['averagePerPlantStats']:
                subdata = subdata.groupby(['Type', 'i']).mean().reset_index()
            
            # Compare every pair of experiments with Mann-Whitney U test
            f.write('Hours from ' + str(step*dt) + ' to ' + str(end) + '\n')
            
            for i in range(0, N_exp-1):
                for j in range(i+1, N_exp):
                    exp1 = subdata[subdata['Type'] == UniqueExperiments[i]]['Signal']
                    exp2 = subdata[subdata['Type'] == UniqueExperiments[j]]['Signal']
                    
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

def growthSpeedsSyncro(frame, N0 = None, N = None, normalize = False, detrend = False, root = 'MainRootLengthGrad (mm/h)', medfilt=False):
    grads_post = []
    dataframes = []
    Ns = []

    for i in range(0, len(frame)):
        data = pd.read_csv(frame[i])
        time = data['ElapsedTime (h)'].to_numpy().astype('int')
        Ns.append(len(time))
        dataframes.append(data)
    
    # Find the minimum N
    N_min = min(Ns)

    for i in range(0, len(frame)):
        data = dataframes[i]

        data['Plant'] = i
        time = data['ElapsedTime (h)'].to_numpy().astype('int')
        newDay = data['NewDay'].to_numpy()

        if N0 is None:
            N0 = 0
        if N is None:
            N = N_min
                
        # remove the first measurements before the new day
        if newDay[N0] != 0:
            begin = np.where(newDay == 0)[0][0]
            if begin > N0:
                N0 = begin
                N = N + begin
            else:
                N0 = N0 + (24 - time[N0])
                N = N + (24 - time[N0])
        
        N = min(N, len(time))
        N = N - N%24

        time = time[N0:N] - N0
        newDay = newDay[N0:N]
        
        mSpeed = data[root].to_numpy()
        
        where_are_NaNs = np.isnan(mSpeed)
        mSpeed[where_are_NaNs] = 0.0

        if normalize:
            mean = np.mean(mSpeed)
            std = np.std(mSpeed)
            if std != 0:
                mSpeed = (mSpeed - mean) / std
            else:
                mSpeed -= mean
            
        where_are_NaNs = np.isnan(mSpeed)
        mSpeed[where_are_NaNs] = 0.0

        if medfilt:
            mSpeed = signal.medfilt(mSpeed, 5)
            mSpeed = mSpeed - signal.medfilt(mSpeed, 25)
        
        if detrend:
            mSpeed = signal.detrend(mSpeed)

        grads_post.append(mSpeed[N0:N])
        
    return grads_post, time, newDay


def process(signal, time, N0, N):
    n = len(signal)
    
    timeb = time - time[0]
    deltat = 1
    deltaf = 1 / (deltat * (N-N0))

    i = 0
    s = signal[i]
    fft = np.abs(np.fft.fft(s))
    fourier = fft
    aux = timeb * deltaf
    
    dataframe = pd.DataFrame(data = {'Time': time, 'Freqs': timeb*deltaf, 'i' : i, 'Signal' : s, 'FFT' : fft})
    
    for i in range(1, n):
        s = signal[i]
        fft = np.abs(np.fft.fft(s))
        fourier = fourier + fft
        aux = pd.DataFrame(data = {'Time': time, 'Freqs': timeb*deltaf, 'i' : i, 'Signal' : s, 'FFT' : fft})
        dataframe = pd.concat([dataframe, aux])
        
    fourier = fourier / n
    
    return dataframe, fourier


def readData(experiments, normalize = False, detrend = False, root = 'MainRootLengthGrad (mm/h)', medfilt=False):
    signals = {}
    times = []

    for exp in experiments:
        plants = load_path(exp, '*/*/*')
        speeds = []

        for plant in plants:
            results = load_path(plant, '*')
            if results == []:
                continue
            else:
                results = results[-1]
            speeds.append(os.path.join(results, "PostProcess_Hour.csv"))

        signal1, time, v = growthSpeedsSyncro(speeds, normalize = normalize, detrend = detrend, root = root, medfilt=medfilt)
        signals[exp] = signal1
        times.append(time)

    N0 = 0
    N = min([len(t) for t in times])
    
    dfs = []
    fouriers = []

    for exp in experiments:
        signal1 = signals[exp]
    
        df, fourier = process(signal1, time, N0, N)
        df['Type'] = exp.split('/')[-1]

        dfs.append(df)
        fouriers.append(fourier)

    return pd.concat(dfs, ignore_index=True), fouriers, time 

def makeFourierPlots(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig3 = plt.figure(figsize=(12,8), constrained_layout=True, dpi = 300)
    gs = fig3.add_gridspec(2, 2)
    f_ax1 = fig3.add_subplot(gs[0, 0])
    f_ax2 = fig3.add_subplot(gs[0, 1])
    f_ax3 = fig3.add_subplot(gs[1, 0])
    f_ax4 = fig3.add_subplot(gs[1, 1])

    all_frames, fouriers, time = readData(experiments)
    
    performStatisticalAnalysis(conf, all_frames, signal = "MR")

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='se', ax=f_ax1, estimator=np.mean)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax1.axvline(j, color = 'green')
            
    f_ax1.set_ylabel('Speed (mm/h)')
    f_ax1.set_xlabel('Time (h)')
    f_ax1.set_title('MR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax1.get_legend_handles_labels()
    f_ax1.legend(handles, labels, loc=2)

    all_frames, _, _ = readData(experiments, root="TotalLengthGrad (mm/h)")

    performStatisticalAnalysis(conf, all_frames, signal = "TR")

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='se', ax=f_ax2, estimator=np.mean)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax2.axvline(j, color = 'green')
            
    f_ax2.set_ylabel('Speed (mm/h)')
    f_ax2.set_xlabel('Time (h)')
    f_ax2.set_title('TR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax2.get_legend_handles_labels()
    f_ax2.legend(handles, labels, loc=2)

    all_frames, fouriers, _ = readData(experiments, normalize = True, medfilt = True)
    
    timeb = time - time[0]
    
    deltat = 1
    deltaf = 1 / (deltat * len(time))

    peak_12 = 0
    peak_24 = 0

    for i in range(0, len(timeb)):
        freq = timeb[i]*deltaf
        
        if np.abs(freq - 1/24) < 0.0001:
            peak_24 = fouriers[0][i]
            
        if np.abs(freq - 1/12) < 0.0001:
            peak_12 = fouriers[0][i]

    sns.lineplot(x = 'Freqs', y = 'FFT', hue = 'Type', data = all_frames, errorbar = 'se', ax=f_ax3)

    f_ax3.axvline(x = 1/24, ymin = 0, ymax = peak_24/25, color = 'red')
    f_ax3.axvline(x = 1/12, ymin = 0, ymax = peak_12/25, color = 'black')

    f_ax3.set_xlim(0, 0.5)
    f_ax3.set_ylim(0, 25)

    f_ax3.set_title('Fourier Transform', fontsize = 16)
    f_ax3.set_xlabel('Frequency (1/hour)')
    f_ax3.set_ylabel('Energy')

    exp = 1.75 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
    exp2 = 1.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)

    sns.lineplot(x="Time", y = "Signal", data = all_frames, hue="Type", errorbar='se', ax=f_ax4)
    for j in range(0, len(time)):
        if j % 24 == 0:
            f_ax4.axvline(j, color = 'green')
            
    f_ax4.set_ylabel('Speed (normalized)')
    f_ax4.set_xlabel('Time (h)')
    f_ax4.set_title('MR Growth Speed', fontsize = 16)
    handles, labels = ax=f_ax4.get_legend_handles_labels()
    f_ax4.legend(handles, labels, loc=4)

    f_ax4.plot(time, exp, color = 'red')
    f_ax4.plot(time, exp2, color = 'black')

    plt.savefig(os.path.join(reportPath_fourier, "JointPlot.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(reportPath_fourier, "JointPlot.svg"), dpi=300, bbox_inches='tight')
    
    makeIndividualFourierPlotsMR(conf)
    makeIndividualFourierPlotsTR(conf)
    makeIndividualFourierPlotsMR_Norm(conf)
    makeIndividualFourier(conf)
           

def makeIndividualFourierPlotsMR(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    all_frames, _, time = readData(experiments)
    
    UniqueExperiments = all_frames['Type'].unique()
    N_exp = int(len(UniqueExperiments))
    
    plt.subplots(1, N_exp, figsize=(6, 3 * N_exp))
    
    min_s, mean_s, sd_s = np.min(all_frames['Signal']), np.mean(all_frames['Signal']), np.std(all_frames['Signal'])
    max_s = mean_s + 3 * sd_s
    
    # Make the plot colors follow the same color scheme as the global plot
    colors = sns.color_palette("tab10", N_exp)
    
    for i in range(0, N_exp):
        subdata = all_frames[all_frames['Type'] == UniqueExperiments[i]]
        ax = plt.subplot(N_exp, 1, i+1)
        
        sns.lineplot(x="Time", y = "Signal", data = subdata, errorbar='se', ax=ax, 
                     estimator=np.mean, color=colors[i])
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax.axvline(j, color = 'green')
                
        ax.set_ylabel('Speed (mm/h)')
        ax.set_ylim(min_s, max_s)
        
        if i == N_exp - 1:
            ax.set_xlabel('Time (h)')
        else:
            ax.set_xlabel('')
            
        # Put legend with the same colors as the global plot
        ax.legend([UniqueExperiments[i]], loc=2)
        
        if i == 0:
            # Create a second x-axis for displaying the hours
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            # Calculate the total number of days
            total_days = np.ceil(subdata['Time'].max() / 24).astype(int)

            # Create day ticks
            day_ticks = np.arange(24, total_days * 24 + 1, 24)

            # Set day ticks and labels
            ax2.set_xticks(day_ticks)
            ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)], rotation=45)
            ax2.tick_params(axis='x', which='major', labelsize=12)
            ax2.tick_params(axis='x', which='major', length=0)
    
    plt.suptitle('MR Growth Speed', fontsize = 16)

    plt.savefig(os.path.join(reportPath_fourier, "MR_Speed.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(reportPath_fourier, "MR_Speed.svg"), dpi=300, bbox_inches='tight')


def makeIndividualFourierPlotsTR(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    all_frames, _, time = readData(experiments, root="TotalLengthGrad (mm/h)")
    
    UniqueExperiments = all_frames['Type'].unique()
    N_exp = int(len(UniqueExperiments))
    
    plt.subplots(1, N_exp, figsize=(6, 3 * N_exp))
    
    min_s, mean_s, sd_s = np.min(all_frames['Signal']), np.mean(all_frames['Signal']), np.std(all_frames['Signal'])
    max_s = mean_s + 4 * sd_s
    
    # Make the plot colors follow the same color scheme as the global plot
    colors = sns.color_palette("tab10", N_exp)
    
    for i in range(0, N_exp):
        subdata = all_frames[all_frames['Type'] == UniqueExperiments[i]]
        ax = plt.subplot(N_exp, 1, i+1)
        
        sns.lineplot(x="Time", y = "Signal", data = subdata, errorbar='se', ax=ax, 
                     estimator=np.mean, color=colors[i])
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax.axvline(j, color = 'green')
                
        ax.set_ylabel('Speed (mm/h)')
        ax.set_ylim(min_s, max_s)
        ax.legend([UniqueExperiments[i]], loc=2)
        
        if i == N_exp - 1:
            ax.set_xlabel('Time (h)')
        else:
            ax.set_xlabel('')
        
        if i == 0:
            # Create a second x-axis for displaying the hours
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            # Calculate the total number of days
            total_days = np.ceil(subdata['Time'].max() / 24).astype(int)

            # Create day ticks
            day_ticks = np.arange(24, total_days * 24 + 1, 24)

            # Set day ticks and labels
            ax2.set_xticks(day_ticks)
            ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)], rotation=45)
            ax2.tick_params(axis='x', which='major', labelsize=12)
    
    plt.suptitle('TR Growth Speed', fontsize = 16)

    plt.savefig(os.path.join(reportPath_fourier, "TR_Speed.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(reportPath_fourier, "TR_Speed.svg"), dpi=300, bbox_inches='tight')
    

def makeIndividualFourierPlotsMR_Norm(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    all_frames, _, time = readData(experiments, normalize = True, medfilt=True)

    # Export the data to a csv file
    all_frames.to_csv(os.path.join(reportPath_fourier, "MR_Norm_Speed.csv"), index=False)
    
    UniqueExperiments = all_frames['Type'].unique()
    N_exp = int(len(UniqueExperiments))
    
    plt.subplots(1, N_exp + 1, figsize=(6, 3 * N_exp))
        
    # Make the plot colors follow the same color scheme as the global plot
    colors = sns.color_palette("tab10", N_exp)
        
    for i in range(0, N_exp):
        subdata = all_frames[all_frames['Type'] == UniqueExperiments[i]]
        ax = plt.subplot(N_exp + 1, 1, i+1)
        
        sns.lineplot(x="Time", y = "Signal", data = subdata, errorbar='se', ax=ax, 
                     estimator=np.mean, color=colors[i])
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax.axvline(j, color = 'green')
                
        ax.set_ylabel('Speed (mm/h)')
        ax.set_ylim(-1, 1)
        ax.legend([UniqueExperiments[i]], loc=2)
        
        if i == N_exp - 1:
            ax.set_xlabel('Time (h)')
        else:
            ax.set_xlabel('')
            
        if i == 0:
            # Create a second x-axis for displaying the hours
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            # Calculate the total number of days
            total_days = np.ceil(subdata['Time'].max() / 24).astype(int)

            # Create day ticks
            day_ticks = np.arange(24, total_days * 24 + 1, 24)

            # Set day ticks and labels
            ax2.set_xticks(day_ticks)
            ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)], rotation=45)
            ax2.tick_params(axis='x', which='major', labelsize=12)
    
    ax = plt.subplot(N_exp + 1, 1, i+2)
    
    exp = 0.5 - 0.45 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
    exp2 = -0.5 + 0.45 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
    
    ax.plot(time, exp, color = 'red')
    ax.plot(time, exp2, color = 'black')
    ax.legend(['24h', '12h'], loc=4)
    ax.set_ylabel('Reference sinusoids')
    ax.set_ylim(-1, 1)
    
    plt.suptitle('MR Growth Speed (Normalized with Median Filter)', fontsize = 16)

    plt.savefig(os.path.join(reportPath_fourier, "MR_Norm_Speed.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(reportPath_fourier, "MR_Norm_Speed.svg"), dpi=300, bbox_inches='tight')
    
    
def makeIndividualFourier(conf):
    analysis = os.path.join(conf['MainFolder'],'Analysis')
    experiments = load_path(analysis, '*')
    
    reportPath = os.path.join(conf['MainFolder'],'Report')
    # Fourier analysis of the growth
    reportPath_fourier = os.path.join(reportPath, 'GrowthSpeeds and Fourier')
    
    if not os.path.exists(reportPath_fourier):
        os.makedirs(reportPath_fourier)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    all_frames, fouriers, time = readData(experiments, normalize = True, medfilt = True)

    timeb = time - time[0]
    
    deltat = 1
    deltaf = 1 / (deltat * len(time))
            
    UniqueExperiments = all_frames['Type'].unique()
    N_exp = int(len(UniqueExperiments))
    
    plt.subplots(1, N_exp, figsize=(6, 3 * N_exp))
    
    min_s, max_s = np.min(all_frames['FFT']), np.max(all_frames['FFT']), 
    
    max_s = 30
    
    # Make the plot colors follow the same color scheme as the global plot
    colors = sns.color_palette("tab10", N_exp)
    
    for j in range(0, N_exp):
        subdata = all_frames[all_frames['Type'] == UniqueExperiments[j]]
        ax = plt.subplot(N_exp, 1, j+1)
        
        sns.lineplot(x="Freqs", y = "FFT", data = subdata, errorbar='se', ax=ax, 
                     estimator=np.mean, color=colors[j])
        
        peak_12_pos = 0
        peak_24_pos = 0
        
        for i in range(0, len(timeb)):
            freq = timeb[i]*deltaf
            
            if np.abs(freq - 1/24) < 0.001:
                peak_24_pos = i
                
            if np.abs(freq - 1/12) < 0.001:
                peak_12_pos = i

        peak_12 = subdata.groupby('Freqs').mean()['FFT'].iloc[peak_12_pos]
        peak_24 = subdata.groupby('Freqs').mean()['FFT'].iloc[peak_24_pos]
        
        # Percentaje of total energy
        peak_12_over_all = round(peak_12 / subdata.groupby('Freqs').mean()['FFT'].sum() * 100, 2)
        peak_24_over_all = round(peak_24 / subdata.groupby('Freqs').mean()['FFT'].sum() * 100, 2)
        peak_24_over_12 = round(peak_24 / peak_12, 2)
        peak_12_over_24 = round(peak_12 / peak_24, 2)
        
        # how do i add a % sign to the text?
        ax.text(0.35, 0.9, 'Percentage of 1/12h over sum: %.2f%%' % peak_12_over_all, fontsize=10, transform=ax.transAxes)
        ax.text(0.35, 0.8, 'Percentage of 1/24h over sum: %.2f%%' % peak_24_over_all, fontsize=10, transform=ax.transAxes)
        ax.text(0.35, 0.7, '1/24h over 1/12h: %.2f' % peak_24_over_12, fontsize=10, transform=ax.transAxes)
        ax.text(0.35, 0.6, '1/12h over 1/24h: %.2f' % peak_12_over_24, fontsize=10, transform=ax.transAxes)
        
        ax.axvline(x = 1/24, ymin = 0, ymax = peak_24/max_s, color = 'red')
        ax.axvline(x = 1/12, ymin = 0, ymax = peak_12/max_s, color = 'black')
                
        ax.set_ylabel('Energy')
        ax.set_ylim(min_s, max_s)
        ax.set_xlim(0, 0.5)
        ax.legend([UniqueExperiments[j]], loc=4)
        
        if i == N_exp - 1:
            ax.set_xlabel('Frequency (1/hour)')
        else:
            ax.set_xlabel('')
    
    plt.suptitle('MR Normalized Filtered - Fourier Transform', fontsize = 16)

    plt.savefig(os.path.join(reportPath_fourier, "MR_Fourier.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(reportPath_fourier, "MR_Fourier.svg"), dpi=300, bbox_inches='tight')
