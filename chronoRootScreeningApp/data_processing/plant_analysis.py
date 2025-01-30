import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from data_processing.fourier import process_data, plot_fourier_summary, store_results

class PlantGrowthAnalyzer:
    """Analyzes multiple plant growth parameters and generates plots."""
    
    PARAMETERS = [
        "HypocotylLength", "Area", "MainRootLength", "TotalRootLength", "DenseRootArea"
    ]
    
    def __init__(self, data=None, output_dir='.', dt=0.25):
        """
        Initialize the analyzer.
        
        Args:
            data: DataFrame with required columns (ElapsedHours, [Parameters], Group, UID)
            output_dir: Directory for saving outputs
            dt: Time between measurements in hours (default 0.25 = 15 minutes)
        """
        self.data = data
        self.base_dir = output_dir
        self.dt = dt
        self.plot_dirs = {}
        
    def _setup_directories(self, parameter):
        """Create output directories for a specific parameter."""
        dirs = {
            'plots': os.path.join(self.base_dir, parameter, f'{parameter}Plots'),
            'fourier': os.path.join(self.base_dir, parameter, 'FourierAnalysis'),
            'data': os.path.join(self.base_dir, parameter, 'ProcessedData')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
    
    def analyze_all_parameters(self):
        """Run complete analysis for all parameters."""
        results = {}
        
        for parameter in self.PARAMETERS:
            print(f"\nAnalyzing {parameter}...")
            self.plot_dirs = self._setup_directories(parameter)
            try:
                results[parameter] = self.analyze_parameter(parameter)
            except Exception as e:
                print(f"Error analyzing {parameter}: {str(e)}")
                continue
        
        return results
    
    def analyze_parameter(self, parameter):
        """Run complete analysis for a single parameter."""
        cleaned_data_list = []
        
        # Clean and validate data
        # First get the lenghts of the different time series
        lengths = []
        for uid in self.data['UID'].unique():
            uid_data = self.data[self.data['UID'] == uid]
            lengths.append(len(uid_data))

        for uid in self.data['UID'].unique():
            uid_data = self.data[self.data['UID'] == uid]
            is_valid, cleaned_series = self._validate_time_series(uid_data, parameter, lengths)
            if is_valid:
                cleaned_data_list.append(cleaned_series)
        
        if not cleaned_data_list:
            raise ValueError(f"No valid time series found for {parameter}")
        
        # Combine cleaned data
        clean_data = pd.concat(cleaned_data_list, ignore_index=True)
        
        # Save cleaned raw data
        clean_data.to_csv(
            os.path.join(self.plot_dirs['data'], f'CleanedTimeSeries_{parameter}.tsv'),
            sep='\t', index=False
        )
        
        # Process hourly data
        processed_data = self._process_to_hourly(clean_data, parameter)
        
        # Save hourly data
        processed_data.to_csv(
            os.path.join(self.plot_dirs['data'], f'HourlyData_{parameter}.tsv'),
            sep='\t', index=False
        )
        
        # Generate all plots
        self._create_all_plots(processed_data, parameter)

        # Perform Fourier analysis
        data, fft = process_data(processed_data, 'GrowthRate', 'SyncHour')
        plot_fourier_summary(data, fft, savepath=self.plot_dirs['fourier'])
        store_results(self.plot_dirs['fourier'], data, fft)

        return {
            'CleanedData': clean_data,
            'HourlyData': processed_data
        }
    
    def _validate_time_series(self, uid_data, parameter, lengths):
        """Validates a single time series for a specific parameter."""

        # Check if the length of the time series is at least 2/3 of the maximum length
        if len(uid_data) < 2/3 * max(lengths):
            return False, None
        
        elapsed_hours = uid_data['ElapsedHours'].values
        parameter_values = uid_data[parameter].values
        full_date = uid_data['Date'].values
        
        # Basic validation checks
        if max(parameter_values) < 0.1 and 'Length' in parameter:
            return False, None
            
        # Enforce non-decreasing values (if appropriate for the parameter)
        # Note: This might need to be adjusted for some parameters
        for i in range(1, len(parameter_values)):
            if parameter_values[i] < parameter_values[i-1]:
                parameter_values[i] = parameter_values[i-1]
        
        # Remove abrupt outliers in growth speed
        if len(parameter_values) > 1:
            speed = np.diff(parameter_values)
            mean_speed = np.mean(speed)
            std_speed = np.std(speed)
            if np.any(speed > (mean_speed + 10 * std_speed)):
                return False, None
        
        # Perform a smoothing on the signal
        parameter_values = signal.medfilt(parameter_values, 5)
        
        # Process date information
        hours = [int(date.split(' ')[1].split('-')[0]) for date in full_date]
        minutes = [int(date.split(' ')[1].split('-')[1]) for date in full_date]
        hours = np.array(hours)
        minutes = np.array(minutes)
        
        # Calculate day changes
        day_change = np.diff(hours) < 0
        
        return True, pd.DataFrame({
            'ElapsedHours': elapsed_hours,
            parameter: parameter_values,
            'Date': full_date,
            'DayChange': np.concatenate([[False], day_change]),
            'Hour': hours,
            'Minute': minutes,
            'Group': uid_data['Group'].iloc[0],
            'UID': uid_data['UID'].iloc[0]
        })

    def _process_to_hourly(self, data, parameter):
        """Process data to hourly intervals with synchronized hours based on first 00:00."""
        processed_data_list = []
        
        for uid in data['UID'].unique():
            uid_data = data[data['UID'] == uid].copy()
            
            # Process each day separately
            day_markers = np.where(uid_data['DayChange'])[0]
            day_starts = [0] + list(day_markers + 1)
            day_ends = list(day_markers) + [len(uid_data)]
            
            day_data_list = []
            for start, end in zip(day_starts, day_ends):
                day_data = uid_data.iloc[start:end].copy()
                
                # Group by hour within this day
                hourly_data = day_data.groupby('Hour').agg({
                    parameter: 'mean',
                    'Group': 'first',
                    'UID': 'first',
                    'Date': 'first'
                }).reset_index()
                
                # Add DayChange column after aggregation
                hourly_data['DayChange'] = False
                if len(day_data_list) > 0:  # If not the first day
                    hourly_data.iloc[0, hourly_data.columns.get_loc('DayChange')] = True

                day_data_list.append(hourly_data)
            
            # Combine all days for this UID
            uid_hourly = pd.concat(day_data_list, ignore_index=True)
            
            # Find the first occurrence of hour 0 (midnight)
            first_midnight_idx = uid_hourly[uid_hourly['Hour'] == 0].index[0]
            
            # Create synchronized hour index
            hours_since_midnight = []
            current_offset = -first_midnight_idx
            
            for i in range(len(uid_hourly)):
                hours_since_midnight.append(current_offset)
                if i < len(uid_hourly) - 1:
                    hour_diff = uid_hourly.iloc[i+1]['Hour'] - uid_hourly.iloc[i]['Hour']
                    if hour_diff < 0:  # Day changed
                        current_offset += (24 - uid_hourly.iloc[i]['Hour'] + uid_hourly.iloc[i+1]['Hour'])
                    else:
                        current_offset += hour_diff
            
            uid_hourly['SyncHour'] = hours_since_midnight
            processed_data_list.append(uid_hourly)
        
        # Combine all processed data
        result = pd.concat(processed_data_list, ignore_index=True)
        
        # Sort by UID and SyncHour
        result = result.sort_values(['UID', 'SyncHour'])
        
        # Calculate growth rate per hour
        result['GrowthRate'] = result.groupby('UID')[parameter].diff()
        
        return result
    
    def _create_all_plots(self, data, parameter):
        """Generate all plots for a specific parameter."""
        self._plot_basic_growth(data, parameter)
        self._plot_individual_groups(data, parameter)
        
    def _plot_basic_growth(self, data, parameter):
        """Create basic growth plots for a specific parameter."""

        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.lineplot(
            data=data,
            x='SyncHour',
            y=parameter,
            hue='Group',
            errorbar='se'
        )
        
        # Add vertical lines every DayChange
        for idx in data[data['DayChange']].index:
            plt.axvline(data.loc[idx, 'SyncHour'], color='green', alpha=0.3, linestyle='--')
        
        plt.title(f'{parameter} Growth')
        plt.xlabel('Time (hours) \n (0 = first midnight)')
        plt.ylabel(f'{parameter} %s'% ('mm' if 'Length' in parameter else 'mm^2'))
        plt.grid(True, alpha=0.3)
    
        ax = plt.subplot(122)
        sns.lineplot(
            data=data,
            x='SyncHour',
            y='GrowthRate',
            hue='Group',
            errorbar='se',
            ax=ax
        )
        
        # Add vertical lines every DayChange
        for idx in data[data['DayChange']].index:
            plt.axvline(data.loc[idx, 'SyncHour'], color='green', alpha=0.3, linestyle='--')
        
        ax.set_title(f'{parameter} Growth Rate')
        plt.xlabel('Time (hours) \n (0 = first midnight)')
        ax.set_ylabel(f'{parameter} Change Rate (%s/hour)'% ('mm' if 'Length' in parameter else 'mm^2'))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dirs['plots'], f'Summarized_{parameter}')
        self._save_plot(plot_path)
    
    def _plot_individual_groups(self, data, parameter):
        """Create individual plots for each group."""
        for group in data['Group'].unique():
            plt.figure(figsize=(15, 10))
            group_data = data[data['Group'] == group]
            
            # Plot parameter values
            plt.subplot(211)
            sns.lineplot(
                data=group_data,
                x='SyncHour',
                y=parameter,
                errorbar='se'
            )
            
            plt.title(f'{group} - {parameter}')
            plt.xlabel('Time (hours)')
            plt.ylabel(f'{parameter} (%s)' % ('mm' if 'Length' in parameter else 'mm^2'))
            plt.grid(True, alpha=0.3)
            
            # Add vertical lines every DayChange
            for idx in group_data[group_data['DayChange']].index:
                plt.axvline(group_data.loc[idx, 'SyncHour'], color='green', alpha=0.3, linestyle='--')
            
            # Plot all individual time series
            plt.subplot(212)
            sns.lineplot(
                data=group_data,
                x='SyncHour',
                y=parameter,
                hue='UID',
                errorbar=None
            )
            
            plt.title(f'{group} - {parameter} (Individual Series)')
            plt.xlabel('Time (hours)')
            plt.ylabel(f'{parameter} (%s)' % ('mm' if 'Length' in parameter else 'mm^2'))
            plt.grid(True, alpha=0.3)
            plt.legend([],[], frameon=False)
            
            # Add vertical lines every DayChange
            for idx in group_data[group_data['DayChange']].index:
                plt.axvline(group_data.loc[idx, 'SyncHour'], color='green', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            plot_path = os.path.join(self.plot_dirs['plots'], f'{parameter}_{group}')
            self._save_plot(plot_path)
    
    def _save_plot(self, name, formats=('.png', '.pdf')):
        """Save plot in multiple formats."""
        for fmt in formats:
            plt.savefig(name + fmt, bbox_inches='tight', dpi=300)
        plt.close()