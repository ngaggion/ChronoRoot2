from .report import load_path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import signal
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
from .utils.fileUtilities import convertFromPathSafe

class MetricConfig:
    """Configuration class for different metrics"""
    METRICS = {
        'MR': {
            'column': 'MainRootLengthGrad (mm/h)',
            'title': 'Main Root Growth Speed',
            'ylabel': 'Speed (mm/h)',
            'norm_ylabel': 'Normalized Speed'
        },
        'TR': {
            'column': 'TotalLengthGrad (mm/h)',
            'title': 'Total Root Growth Speed',
            'ylabel': 'Speed (mm/h)',
            'norm_ylabel': 'Normalized Speed'
        }
    }

    @classmethod
    def get_config(cls, metric_type: str) -> dict:
        """Get configuration for a specific metric"""
        if metric_type not in cls.METRICS:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        return cls.METRICS[metric_type]

class DataProcessor:
    """Class for processing and analyzing growth data"""
    
    def __init__(self, conf: dict):
        self.conf = conf
        self.report_path = os.path.join(conf['MainFolder'], 'Report')
        self.fourier_path = os.path.join(self.report_path, 'GrowthSpeeds and Fourier')
        os.makedirs(self.fourier_path, exist_ok=True)

    def process_single_file(self, filepath: str, N0: Optional[int] = None, 
                          N: Optional[int] = None, root: str = 'MainRootLengthGrad (mm/h)',
                          normalize: bool = False, detrend: bool = False, 
                          medfilt: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single data file"""
        try:
            data = pd.read_csv(filepath)
            time = data['ElapsedTime (h)'].to_numpy().astype('int')
            newDay = data['NewDay'].to_numpy()

            # Handle time range selection
            N0 = 0 if N0 is None else N0
            N = len(time) if N is None else N
            
            # Adjust for new day
            if not np.isscalar(newDay[N0]) or newDay[N0] != 0:
                begin_indices = np.where(newDay == 0)[0]
                if len(begin_indices) > 0:
                    begin = begin_indices[0]
                    if begin > N0:
                        # Shift start point
                        N0 = begin
                        # We do NOT shift N here because we are limited by file length
                    else:
                        N0 = N0 + (24 - time[N0])
            
            # Ensure we don't go past the end of the file
            max_len = len(time)
            if N > max_len:
                N = max_len

            # 1. Calculate available duration
            current_duration = N - N0
            
            # 2. Ensure duration is non-negative
            if current_duration < 0:
                current_duration = 0
                
            # 3. Make the DURATION a multiple of 24, not the index
            valid_duration = current_duration - (current_duration % 24)
            
            # 4. Set new end index based on start + valid duration
            N = N0 + valid_duration

            # Extract and process speed data
            mSpeed = np.array(data[root], copy=True)
            mSpeed = np.nan_to_num(mSpeed, 0.0)

            if normalize:
                mean = np.mean(mSpeed)
                std = np.std(mSpeed)
                mSpeed = (mSpeed - mean) / std if std != 0 else mSpeed - mean

            if medfilt:
                mSpeed = signal.medfilt(mSpeed, 5)
                mSpeed = mSpeed - signal.medfilt(mSpeed, 25)

            if detrend:
                mSpeed = signal.detrend(mSpeed)

            return mSpeed[N0:N], time[N0:N] - N0, newDay[N0:N]
        
        except Exception as e:
            raise Exception(f"Error processing file {filepath}: {str(e)}")

    def read_and_process_data(self, experiments: List[str], root: str = 'MainRootLengthGrad (mm/h)',
                             normalize: bool = False, detrend: bool = False, 
                             medfilt: bool = False) -> pd.DataFrame:
        """Read and process data from multiple experiments with path-safe naming"""
        all_data = []
        valid_datasets = [] # Store dict (exp_label, signal, time)
        
        # 1. Collect all valid processed data first
        for exp in experiments:
            try:
                # Get list of plant data files
                plants = load_path(exp, '*/*/*')
                raw_folder_name = os.path.basename(exp)
                exp_label = convertFromPathSafe(raw_folder_name)
                speeds = []
                
                # Collect PostProcess_Hour.csv files
                for plant in plants:
                    results = load_path(plant, '*')
                    if results:
                        results = results[-1]
                        speeds.append(os.path.join(results, "PostProcess_Hour.csv"))

                for speed_file in speeds:
                    try:
                        # Process individual file
                        signal_arr, time_arr, _ = self.process_single_file(
                            speed_file, 
                            root=root,
                            normalize=normalize,
                            detrend=detrend,
                            medfilt=medfilt
                        )
                        
                        valid_datasets.append({
                            'exp': exp_label, # Use the sanitized label here
                            'file': speed_file,
                            'signal': signal_arr,
                            'time': time_arr
                        })
                        
                    except Exception as e:
                        print(f"Skipping file {speed_file}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing experiment {exp}: {str(e)}")
                continue

        if not valid_datasets:
            raise ValueError("No valid data processed from any experiment")

        # 2. Find the Common Denominator Length
        lengths = [len(d['signal']) for d in valid_datasets]
        min_common_length = min(lengths)
        
        # Ensure it's still a multiple of 24
        min_common_length = min_common_length - (min_common_length % 24)
        
        # 3. Truncate all datasets to the minimum common length
        time_series = None
        
        for item in valid_datasets:
            # Slice to common length
            sig = item['signal'][:min_common_length]
            t = item['time'][:min_common_length]
            
            if time_series is None:
                time_series = t
            
            # Create DataFrame
            df = pd.DataFrame({
                'Time': time_series, # Use uniform time vector
                'Signal': sig,
                'Type': item['exp'], # This now uses the safe label
                'i': len(all_data)
            })
            all_data.append(df)

        if not all_data:
            raise ValueError("No valid processed data after truncation")

        combined_df = pd.concat(all_data, ignore_index=True)

        # 4. Calculate FFT (Now safe because lengths are identical)
        time_length = min_common_length
        freqs = np.fft.fftfreq(time_length, d=1)
        
        fft_data = []
        for (exp_type, i), group in combined_df.groupby(['Type', 'i']):
            signal_data = group['Signal'].values
            fft_vals = np.abs(np.fft.fft(signal_data))
            fft_df = pd.DataFrame({
                'Freqs': freqs,
                'FFT': fft_vals,
                'Type': exp_type,
                'i': i
            })
            fft_data.append(fft_df)
        
        if fft_data:
            fft_combined = pd.concat(fft_data, ignore_index=True)
            combined_df = pd.concat([combined_df, fft_combined], ignore_index=True)

        return combined_df
    
    def perform_statistical_analysis(self, data_orig: pd.DataFrame, data_detrended: pd.DataFrame, metric_type: str):
        """Perform statistical analysis on temporal and frequency data"""
        try:
            stats_path = os.path.join(self.fourier_path, f'{metric_type}_Stats.txt')
            unique_experiments = data_orig['Type'].unique()
            N_exp = len(unique_experiments)

            with open(stats_path, 'w') as f:
                f.write(f'CHRONOROOT 2.0 STATISTICAL REPORT - {metric_type}\n')
                f.write('='*60 + '\n')
                
                # PART 1: Temporal Growth Speed (Original Scale)
                f.write('PART 1: HOURLY GROWTH SPEED COMPARISONS (Original Data)\n')
                dt = int(self.conf['everyXhourFieldFourier'])
                time_data = data_orig[data_orig['Time'].notna()]
                N_steps = int(round((time_data['Time'].max()+1) / dt, 0))

                for step in range(N_steps):
                    end = min(dt * (step+1), time_data['Time'].max())
                    subdata = time_data[time_data['Time'].isin(np.arange(dt * step, end))]
                    if self.conf.get('averagePerPlantStats', False):
                        subdata = subdata.groupby(['Type', 'i']).mean().reset_index()

                    f.write(f'\nWindow: {step*dt}h to {end}h\n')
                    for i in range(N_exp-1):
                        for j in range(i+1, N_exp):
                            self._write_comparison_stats(f, subdata, unique_experiments[i], unique_experiments[j], col='Signal')

                # PART 2: FFT Energy Analysis (Detrended & Normalized)
                f.write('\n' + '='*60 + '\n')
                f.write('PART 2: CIRCADIAN RHYTHM ANALYSIS (Detrended/Normalized FFT)\n')
                f.write('This section compares the oscillatory power after removing growth trends.\n')
                f.write('='*60 + '\n')
                
                fft_detrended = data_detrended[data_detrended['Freqs'].notna()]
                target_rhythms = {'24h Period': 1/24, '12h Period': 1/12}
                
                for label, target_freq in target_rhythms.items():
                    f.write(f'\nFrequency Bin: {label} ({target_freq:.4f} Hz)\n')
                    
                    available_freqs = fft_detrended['Freqs'].unique()
                    closest_freq = available_freqs[np.argmin(np.abs(available_freqs - target_freq))]
                    freq_subdata = fft_detrended[fft_detrended['Freqs'] == closest_freq]
                    
                    for i in range(N_exp-1):
                        for j in range(i+1, N_exp):
                            self._write_comparison_stats(f, freq_subdata, 
                                                       unique_experiments[i], 
                                                       unique_experiments[j], 
                                                       col='FFT', 
                                                       is_fft=True)

        except Exception as e:
            print(f"Error in statistical analysis: {str(e)}")


    def _write_comparison_stats(self, f, subdata: pd.DataFrame, exp1_name: str, exp2_name: str, col='Signal', is_fft=False):
        """Write comparison statistics between two experiments"""
        exp1 = subdata[subdata['Type'] == exp1_name][col]
        exp2 = subdata[subdata['Type'] == exp2_name][col]

        try:
            if len(exp1) == 0 or len(exp2) == 0:
                return

            U, p = stats.mannwhitneyu(exp1, exp2)
            p_val = round(p, 6)

            f.write(f'Comparison: {exp1_name} vs {exp2_name}\n')
            f.write(f'  - Samples: {len(exp1)} vs {len(exp2)}\n')
            f.write(f'  - Mean: {exp1.mean():.4f} vs {exp2.mean():.4f}\n')
            f.write(f'  - Std Dev: {exp1.std():.4f} vs {exp2.std():.4f}\n')
            # Standardized significance notation for the report
            sig_text = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
            stars = "**" if p < 0.001 else ("*" if p < 0.05 else "ns")
            
            metric_label = "FFT Energy" if is_fft else "Speed"
            f.write(f'  - Result: {metric_label} is {sig_text} (p={p_val}, {stars})\n')

        except Exception as e:
            f.write(f'Error comparing {exp1_name} and {exp2_name}: {str(e)}\n')

class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self, fourier_path: str):
        self.fourier_path = fourier_path
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Set up matplotlib plot style"""
        SMALL_SIZE = 10
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

    def create_joint_plot(self, data: pd.DataFrame, data_detrended: pd.DataFrame, 
                         time: np.ndarray, metric_config: dict, 
                         output_prefix: str):
        """Create joint plot with original and detrended data"""
        fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(2, 2)
        
        axes = {
            'original': fig.add_subplot(gs[0, 0]),
            'detrended': fig.add_subplot(gs[0, 1]),
            'fft_original': fig.add_subplot(gs[1, 0]),
            'fft_detrended': fig.add_subplot(gs[1, 1])
        }

        # Plot original data
        self._plot_time_series(axes['original'], data, time, 
                             metric_config['ylabel'], 
                             f"Original {metric_config['title']}")
        
        # Plot detrended data with reference curves
        self._plot_time_series(axes['detrended'], data_detrended, time,
                             metric_config['norm_ylabel'],
                             f"Normalized & Detrended {metric_config['title']}")
        
        # Add reference curves to detrended plot
        exp1 = 1.75 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = 1.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        axes['detrended'].plot(time, exp1, color='red', label='24h rhythm')
        axes['detrended'].plot(time, exp2, color='black', label='12h rhythm')
        
        # Update legend for detrended plot
        handles, labels = axes['detrended'].get_legend_handles_labels()
        axes['detrended'].legend(handles, labels, loc='best')

        # Plot FFTs
        self._plot_fft(axes['fft_original'], data, "FFT of Original Signal")
        self._plot_fft(axes['fft_detrended'], data_detrended, "FFT of Normalized & Detrended Signal")

        plt.suptitle(f"Joint Plot - {metric_config['title']}", fontsize=16, y=1.02)

        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.fourier_path, f"JointPlot_{metric_config['title']}.{ext}"),
                       dpi=300, bbox_inches='tight')
        plt.close()


    def create_individual_plots(self, data: pd.DataFrame, data_detrended: pd.DataFrame,
                                time: np.ndarray, metric_config: dict):
        """Create individual plots for each experiment"""
        unique_experiments = data['Type'].unique()
        n_exp = len(unique_experiments)
        
        # Get color palette for consistent colors across both plots
        colors = sns.color_palette("tab10", n_exp)
        
        # Calculate y-axis limits for original data
        min_signal = data['Signal'].min()
        max_signal = data['Signal'].mean() + 3 * data['Signal'].std()
        
        # First Figure: Original Signals
        fig1 = plt.figure(figsize=(10, 3 * (n_exp + 1)), constrained_layout=True)
        gs1 = fig1.add_gridspec(n_exp + 1, 1)
        
        # Plot original data for each experiment
        for i, exp_name in enumerate(unique_experiments):
            ax = fig1.add_subplot(gs1[i, 0])
            exp_data = data[data['Type'] == exp_name]
            
            sns.lineplot(x="Time", y="Signal", data=exp_data,
                        errorbar='se', ax=ax, color=colors[i],
                        estimator=np.mean)
            
            for j in range(0, len(time)):
                if j % 24 == 0:
                    ax.axvline(j, color='green', alpha=1.0, linestyle='--')
            
            ax.set_ylabel(metric_config['ylabel'])
            ax.set_xlabel('')
            ax.set_ylim(min_signal, max_signal)
            ax.legend([f"{exp_name}"], loc='upper left')
            
            # Add day labels on top for first subplot only
            if i == 0:
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                total_days = np.ceil(exp_data['Time'].max() / 24).astype(int)
                day_ticks = np.arange(24, total_days * 24 + 1, 24)
                ax2.set_xticks(day_ticks)
                ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)])
                ax2.tick_params(axis='x', rotation=45)
        
        # Add reference sinusoids for original scale
        ax_sin = fig1.add_subplot(gs1[-1, 0])
        exp1 = -0.25 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = 0.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        ax_sin.plot(time, exp1, color='red', label='24h rhythm')
        ax_sin.plot(time, exp2, color='black', label='12h rhythm')
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax_sin.axvline(j, color='green', alpha=1.0, linestyle='--')
        
        ax_sin.set_ylabel('Reference Patterns')
        ax_sin.set_xlabel('Time (h)')
        ax_sin.legend(loc='upper right')
        
        plt.suptitle(f"{metric_config['title']} Analysis - Original", fontsize=16, y=1.02)
        
        # Save original plots
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.fourier_path,
                                    f"{metric_config['title']}_individual_original.{ext}"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Second Figure: Normalized Signals
        fig2 = plt.figure(figsize=(10, 3 * (n_exp + 1)), constrained_layout=True)
        gs2 = fig2.add_gridspec(n_exp + 1, 1)
        
        # Plot normalized data for each experiment
        for i, exp_name in enumerate(unique_experiments):
            ax = fig2.add_subplot(gs2[i, 0])
            exp_data_norm = data_detrended[data_detrended['Type'] == exp_name]
            
            sns.lineplot(x="Time", y="Signal", data=exp_data_norm,
                        errorbar='se', ax=ax, color=colors[i],
                        estimator=np.mean)
            
            for j in range(0, len(time)):
                if j % 24 == 0:
                    ax.axvline(j, color='green', alpha=1.0, linestyle='--')
            
            ax.set_ylabel(metric_config['norm_ylabel'])
            ax.set_xlabel('')
            ax.set_ylim(-1, 1)
            ax.legend([f"{exp_name}"], loc='upper left')
            
            # Add day labels on top for first subplot only
            if i == 0:
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                total_days = np.ceil(exp_data_norm['Time'].max() / 24).astype(int)
                day_ticks = np.arange(24, total_days * 24 + 1, 24)
                ax2.set_xticks(day_ticks)
                ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)])
                ax2.tick_params(axis='x', rotation=45)
        
        # Add reference sinusoids for normalized scale
        ax_sin = fig2.add_subplot(gs2[-1, 0])
        exp1 = 0.25 - 0.25 * np.cos(1/24 * (time-12) * 2 * np.pi + np.pi)
        exp2 = -0.25 + 0.25 * np.cos(1/12 * (time-12) * 2 * np.pi + np.pi)
        ax_sin.plot(time, exp1, color='red', label='24h rhythm')
        ax_sin.plot(time, exp2, color='black', label='12h rhythm')
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax_sin.axvline(j, color='green', alpha=1.0, linestyle='--')
        
        ax_sin.set_ylabel('Reference Patterns')
        ax_sin.set_xlabel('Time (h)')
        ax_sin.legend(loc='upper right')
        
        plt.suptitle(f"{metric_config['title']} Analysis - Normalized", fontsize=16, y=1.02)
        
        # Save normalized plots
        for ext in ['png', 'svg']:
            plt.savefig(os.path.join(self.fourier_path,
                                    f"{metric_config['title']}_individual_normalized.{ext}"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_series(self, ax, data: pd.DataFrame, time: np.ndarray, 
                         ylabel: str, title: str):
        """Plot time series data"""
        sns.lineplot(x="Time", y="Signal", data=data,
                    hue="Type", errorbar='se', ax=ax, 
                    estimator=np.mean)
        
        for j in range(0, len(time)):
            if j % 24 == 0:
                ax.axvline(j, color='green', alpha=1.0, linestyle='--')
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (h)')
        ax.set_title(title, fontsize=16)

    def _plot_fft(self, ax, data: pd.DataFrame, title: str):
        """Plot FFT data with annotations"""
        sns.lineplot(x='Freqs', y='FFT', hue='Type',
                    data=data[data['Freqs'] >= 0], errorbar='se', ax=ax)
        
        # Add vertical lines for 24h and 12h periods
        periods = {'24h': 1/24, '12h': 1/12}
        colors = {'24h': 'red', '12h': 'black'}
        
        for period, freq in periods.items():
            # Find peak value at this frequency
            freq_data = data[np.abs(data['Freqs'] - freq) < 0.001]
            if not freq_data.empty:
                ax.axvline(x=freq, ymin=0, ymax=ax.get_ylim()[1],
                          color=colors[period], linestyle='--', alpha=0.5,
                          label=f'{period} period')
        
        ax.set_xlim(0, 0.5)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Frequency (1/hour)')
        ax.set_ylabel('Energy')
        
        # Ensure legend includes period markers
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best')

def makeFourierPlots(conf: dict):
    """Main function to create Fourier analysis plots"""
    try:
        processor = DataProcessor(conf)
        visualizer = Visualizer(processor.fourier_path)
        
        # Process each metric
        for metric_type in MetricConfig.METRICS.keys():
            try:
                print(f"Processing {metric_type}")
                metric_config = MetricConfig.get_config(metric_type)
                
                # Get data paths
                analysis_path = os.path.join(conf['MainFolder'], 'Analysis')
                experiments = load_path(analysis_path, '*')
                
                # Process data
                all_frames_original = processor.read_and_process_data(
                    experiments, 
                    root=metric_config['column']
                )
                
                all_frames_detrended = processor.read_and_process_data(
                    experiments,
                    root=metric_config['column'],
                    normalize=True,
                    detrend=True,
                    medfilt=True
                )
                
                # Perform statistical analysis
                processor.perform_statistical_analysis(
                    all_frames_original, 
                    all_frames_detrended,
                    metric_type
                )
                
                # Create visualizations
                time_series = all_frames_original['Time'].unique()
                visualizer.create_joint_plot(
                    all_frames_original,
                    all_frames_detrended,
                    time_series,
                    metric_config,
                    f"JointPlot_{metric_type}"
                )
                
                # Create individual plots - updated call to match function signature
                visualizer.create_individual_plots(
                    all_frames_original,
                    all_frames_detrended,
                    time_series,
                    metric_config
                )
                
            except Exception as e:
                print(f"Skipping metric {metric_type} due to error: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in makeFourierPlots: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    conf = {
        'MainFolder': '/path/to/main/folder',
        'everyXhourFieldFourier': 24,
        'averagePerPlantStats': True
    }
    makeFourierPlots(conf)