import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import subprocess
from data_processing.fourier import process_data, plot_fourier_summary, store_results
from data_processing.fpca import run_fpca_analysis as run_fpca_func
import argparse

class PlantGrowthAnalyzer:
    """Analyzes multiple plant growth parameters and generates plots."""
    
    PARAMETERS = [
        "HypocotylLength", "Area", "MainRootLength", "TotalRootLength", "DenseRootArea"
    ]
    
    def __init__(self, data=None, output_dir='.', add_time_before_photo=0):
        """
        Initialize the analyzer.
        
        Args:
            data: DataFrame with required columns (ElapsedHours, [Parameters], Group, UID)
            output_dir: Directory for saving outputs
            conda_env: Conda environment name for FPCA analysis (default: "FDA")
        """
        self.data = data
        self.base_dir = output_dir
        self.plot_dirs = {}
        self.add_time_before_photo = add_time_before_photo
        self.all_results = {}
        
    def _setup_directories(self, parameter):
        """Create output directories for a specific parameter."""
        dirs = {
            'plots': os.path.join(self.base_dir, parameter, f'{parameter}Plots'),
            'fourier': os.path.join(self.base_dir, parameter, 'FourierAnalysis'),
            'fpca': os.path.join(self.base_dir, parameter, 'FPCAAnalysis'),
            'data': os.path.join(self.base_dir, parameter, 'ProcessedData')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
    
    def analyze_all_parameters(self, run_fpca=True,  
                              fpca_components=2, fpca_normalize=False):
        """
        Run complete analysis for all parameters.
        
        Args:
            run_fpca: Whether to run FPCA on hourly data (default: True)
            fpca_components: Number of FPCA components (default: 2)
            fpca_normalize: Whether to normalize FPCA data (default: False)
            
        Returns:
            Dictionary with analysis results for all parameters
        """
        results = {}
        
        for parameter in self.PARAMETERS:
            print(f"\n{'='*50}")
            print(f"Analyzing {parameter}...")
            print(f"{'='*50}")
            
            self.plot_dirs = self._setup_directories(parameter)
            
            try:
                # Run standard analysis with optional FPCA
                param_results = self.analyze_parameter(
                    parameter,
                    run_fpca=run_fpca,
                    fpca_components=fpca_components,
                    fpca_normalize=fpca_normalize
                )
                                                
                results[parameter] = param_results
                
            except Exception as e:
                print(f"Error analyzing {parameter}: {str(e)}")
                results[parameter] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Store the comprehensive results
        self.all_results = results
                
        return results
    
    def analyze_parameter(self, parameter, run_fpca=True, fpca_components=2, fpca_normalize=False):
        """
        Run complete analysis for a single parameter.
        
        Args:
            parameter: Parameter to analyze (e.g., "HypocotylLength")
            run_fpca: Whether to run FPCA analysis (default: True)
            fpca_components: Number of FPCA components (default: 2)
            fpca_normalize: Whether to normalize FPCA data (default: False)
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'parameter': parameter,
            'status': 'processing'
        }
        
        try:
            cleaned_data_list = []
            
            # Clean and validate data
            # First get the lengths of the different time series
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
            clean_data_path = os.path.join(self.plot_dirs['data'], f'CleanedTimeSeries_{parameter}.tsv')
            clean_data.to_csv(clean_data_path, sep='\t', index=False)
            results['CleanedData'] = {
                'data': clean_data,
                'path': clean_data_path
            }
            
            # Process hourly data
            processed_data = self._process_to_hourly(clean_data, parameter)
            
            # Save hourly data
            hourly_data_path = os.path.join(self.plot_dirs['data'], f'HourlyData_{parameter}.tsv')
            processed_data.to_csv(hourly_data_path, sep='\t', index=False)
            results['HourlyData'] = {
                'data': processed_data,
                'path': hourly_data_path
            }
            
            # Generate all plots
            plot_paths = self._create_all_plots(processed_data, parameter)
            results['Plots'] = plot_paths
    
            # Perform Fourier analysis
            try:
                fourier_data, fft_data = process_data(processed_data, 'GrowthRate', 'SyncHour')
                plot_fourier_summary(fourier_data, fft_data, savepath=self.plot_dirs['fourier'])
                store_results(self.plot_dirs['fourier'], fourier_data, fft_data)
                
                results['FourierData'] = {
                    'normalized_data': fourier_data,
                    'fft_data': fft_data,
                    'path': self.plot_dirs['fourier'],
                    'status': 'success'
                }
            except Exception as e:
                print(f"Error in Fourier analysis for {parameter}: {str(e)}")
                results['FourierData'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Run FPCA analysis if requested
            if run_fpca:
                try:
                    fpca_path = self.run_fpca_analysis(
                        parameter, 
                        components=fpca_components, 
                        normalize=fpca_normalize
                    )
                    results['FPCA'] = {
                        'path': fpca_path,
                        'components': fpca_components,
                        'normalized': fpca_normalize,
                        'status': 'success'
                    }
                except Exception as e:
                    print(f"Error in FPCA analysis for {parameter}: {str(e)}")
                    results['FPCA'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
    
            results['status'] = 'success'
        except Exception as e:
            print(f"Error analyzing {parameter}: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
        return results
    
    def _validate_time_series(self, uid_data, parameter, lengths):
        """Validates a single time series for a specific parameter."""

        # Check if the length of the time series is at least 2/3 of the maximum length
        if len(uid_data) < max(lengths):
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
        """Process data to hourly intervals with synchronized hours based on earliest timestamp."""
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
            
            # Create sequential hour index based on original hours
            sequential_hours = []
            current_offset = 0
            
            for i in range(len(uid_hourly)):
                sequential_hours.append(current_offset)
                if i < len(uid_hourly) - 1:
                    hour_diff = uid_hourly.iloc[i+1]['Hour'] - uid_hourly.iloc[i]['Hour']
                    if hour_diff < 0:  # Day changed
                        current_offset += (24 - uid_hourly.iloc[i]['Hour'] + uid_hourly.iloc[i+1]['Hour'])
                    else:
                        current_offset += hour_diff
            
            uid_hourly['SyncHour'] = sequential_hours
            processed_data_list.append(uid_hourly)
        
        # Combine all processed data
        result = pd.concat(processed_data_list, ignore_index=True)
        
        # Find the minimum SyncHour value to use as reference point (0)
        min_hour = result['SyncHour'].min() - self.add_time_before_photo
        
        # Adjust all SyncHour values so the earliest time is 0
        result['SyncHour'] = result['SyncHour'] - min_hour
        
        # Sort by UID and SyncHour
        result = result.sort_values(['UID', 'SyncHour'])
        
        # Calculate growth rate per hour
        result['GrowthRate'] = result.groupby('UID')[parameter].diff()
        # Fill na values with 0, for the first hour
        result['GrowthRate'] = result['GrowthRate'].fillna(0)
        
        return result
    
    def _create_all_plots(self, data, parameter):
        """
        Generate all plots for a specific parameter.
        
        Returns:
            Dictionary with paths to generated plots
        """
        plot_paths = {}
        
        # Basic growth plots
        basic_path = self._plot_basic_growth(data, parameter)
        plot_paths['basic_growth'] = basic_path
        
        # Group plots
        group_paths = self._plot_individual_groups(data, parameter)
        plot_paths['group_plots'] = group_paths
        
        return plot_paths
        
    def _plot_basic_growth(self, data, parameter):
        """
        Create basic growth plots for a specific parameter.
        
        Returns:
            Path to the generated plot
        """
        data = data.copy()
        data['Group'] = data['Group'].astype(str).str.strip()
        
        plt.figure(figsize=(8, 4))
        
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
        
        # X-ticks should be at the vertical grid lines ad DayChange
        x_ticks = data[data['DayChange']]['SyncHour'].values
        plt.xticks(x_ticks)

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

        # X-ticks should be at the vertical grid lines ad DayChange
        x_ticks = data[data['DayChange']]['SyncHour'].values
        ax.set_xticks(x_ticks)

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dirs['plots'], f'Summarized_{parameter}')
        self._save_plot(plot_path)
        
        return plot_path
    
    def _plot_individual_groups(self, data, parameter):
        """
        Create individual plots for each group.
        
        Returns:
            Dictionary with paths to group plots
        """
        data = data.copy()
        data['Group'] = data['Group'].astype(str).str.strip()
        group_paths = {}
        
        for group in data['Group'].unique():
            plt.figure(figsize=(8, 4))
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
            
            # X-ticks should be at the vertical grid lines ad DayChange
            x_ticks = group_data[group_data['DayChange']]['SyncHour'].values
            plt.xticks(x_ticks)

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
            
            # X-ticks should be at the vertical grid lines ad DayChange
            x_ticks = group_data[group_data['DayChange']]['SyncHour'].values
            plt.xticks(x_ticks)

            plt.tight_layout()
            plot_path = os.path.join(self.plot_dirs['plots'], f'{parameter}_{group}')
            self._save_plot(plot_path)
            
            group_paths[group] = plot_path
            
        return group_paths
    
    def _save_plot(self, name, formats=('.png', '.pdf')):
        """
        Save plot in multiple formats.
        
        Returns:
            Dictionary with paths to saved files
        """
        saved_paths = {}
        for fmt in formats:
            file_path = name + fmt
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            saved_paths[fmt[1:]] = file_path
        plt.close()
        return saved_paths

    def run_fpca_analysis(self, parameter, components=2, normalize=False):
        """
        Run FPCA analysis on hourly data results for a specific parameter.
        
        Args:
            parameter: Plant parameter to analyze
            components: Number of FPCA components (default: 2)
            normalize: Whether to apply inverse rank normalization (default: False)
            
        Returns:
            Path to the directory containing FPCA results
        """
        # Get the hourly data path
        hourly_data_path = os.path.join(
            self.plot_dirs['data'], 
            f'HourlyData_{parameter}.tsv'
        )
        
        # Create output directory for FPCA results
        fpca_output_dir = os.path.join(
            self.base_dir, 
            parameter, 
            'FPCAAnalysis'
        )
        os.makedirs(fpca_output_dir, exist_ok=True)
        
        # Create arguments object for FPCA
        fpca_args = argparse.Namespace(
            file=hourly_data_path,
            xcol="SyncHour",
            ycols=[parameter, "GrowthRate"],
            output=fpca_output_dir,
            normalize=normalize,
            components=components,
            groupby="Group",
            id_col="UID",
            figsize=[8, 16],
            dpi=300
        )
        
        # Run the FPCA analysis
        print(f"Running FPCA analysis for {parameter}...")
        try:
            run_fpca_func(fpca_args)
            print(f"FPCA analysis completed for {parameter}. Results saved to {fpca_output_dir}")
        except Exception as e:
            error_msg = f"Error running FPCA analysis for {parameter}: {str(e)}"
            print(error_msg)
            with open(os.path.join(fpca_output_dir, "error_log.txt"), "w") as f:
                f.write(error_msg)
            raise RuntimeError(error_msg)
        
        return fpca_output_dir