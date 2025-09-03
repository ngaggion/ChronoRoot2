import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.optimize import curve_fit

class GerminationAnalyzer:
    """Analyzes seed germination data and generates plots."""
    
    def __init__(self, data=None, output_dir='.', dt=1, add_time_before_photo=0):
        self.data = self._prepare_data(data)
        self.base_dir = output_dir
        self.plot_dirs = self._setup_directories()
        
        # Analysis parameters
        self.TIME_WINDOW = dt
        self.ROLL_WINDOW = max(5, int(round(2.5 / self.TIME_WINDOW)))
        self.DETECTION_WINDOW = max(10, int(round(5 / self.TIME_WINDOW)))
        self.add_time_before_photo = add_time_before_photo
        
        self.germination_data = None
        self.statistics = None
    
    def _prepare_data(self, data):
        """Prepare input data."""
        if data is None:
            return None
        
        data = data.copy()
        data['Group'] = data['Group'].astype(str)
        data['Group'] = data['Group'].str.replace('/', '_')
        data['UID'] = data['UID'].str.replace('/', '_')
        data['UID'] = data['UID'].astype(str)
        return data[data['Group'] != 'Unknown']
    
    def _setup_directories(self):
        """Create output directories."""
        dirs = {
            'germination': os.path.join(self.base_dir, 'Germination', 'GerminationPlots'),
            'germination_video': os.path.join(self.base_dir, 'Germination', 'GerminationPlotsPerVideo'),
            'survival': os.path.join(self.base_dir, 'Germination', 'KaplanMeierPlots'),
            'data': os.path.join(self.base_dir, 'Germination', 'ProcessedData')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
        
    @staticmethod
    def hill_function(x, y0, max_germ, steepness, t50):
        """Four-parameter Hill function for curve fitting."""
        return y0 + (max_germ * x**steepness) / (t50**steepness + x**steepness)
    
    @staticmethod
    def germination_rate(x, y0, max_germ, steepness, t50):
        """Calculate instantaneous germination rate."""
        return (max_germ * steepness * t50**steepness * x**(steepness-1)) / \
               (t50**steepness + x**steepness)**2
    
    @staticmethod
    def find_TMGR(params):
        """Calculate Time of Maximum Germination Rate."""
        _, _, steepness, t50 = params
        return t50 * ((steepness - 1)/(steepness + 1))**(1/steepness)
    
    def analyze(self):
        """Run complete germination analysis."""
        self._process_raw_data()
        self._calculate_statistics()
        self._create_plots()
        return {
            'ProcessedData': self.data,
            'GerminationData': self.germination_data,
            'Statistics': self.statistics
        }
    
    def _detect_germination(self, group):
        """Detect germination events based on growth patterns."""
        group['GrowthRate'] = (group['Perim.']
                             .rolling(window=self.ROLL_WINDOW, min_periods=1)
                             .mean() - group['Perim.'].mean())
        group['RateChange'] = group['GrowthRate'].diff().fillna(0)
        group['ShortTermChange'] = group['GrowthRate'].diff(periods=self.DETECTION_WINDOW)
        group['LongTermChange'] = group['GrowthRate'].diff(periods=self.DETECTION_WINDOW * 2)
        group['TrendDirection'] = (group['RateChange']
                                 .apply(np.sign)
                                 .rolling(window=self.DETECTION_WINDOW, min_periods=1)
                                 .mean())
        
        group['Germinated'] = ((group['TrendDirection'] >= 0.8) & 
                              (group['ShortTermChange'] > 0.05) & 
                              (group['LongTermChange'] > 0.1))
        return group

    def _process_raw_data(self):
        """Process raw measurements to detect germination events."""
        data = self.data.copy()
        
        # Calculate germination metrics
        data = data.groupby('UID', group_keys=False).apply(self._detect_germination)
        data['ElapsedHours'] = data['ElapsedHours'] + self.add_time_before_photo
        
        # Get unique combinations of Group and Video
        group_video_counts = (data[['Group', 'Video', 'SeedCount']]
                            .drop_duplicates()
                            .set_index(['Group', 'Video'])['SeedCount']
                            .to_dict())
        
        # Summarize germination events
        germ_data = (data[data['Germinated']]
                    .groupby(['Group', 'UID', 'Video'])
                    .agg({
                        'ElapsedHours': 'min',
                        'Area': 'mean'
                    })
                    .reset_index()
                    .rename(columns={
                        'ElapsedHours': 'GerminationTime',
                        'Area': 'SeedSize'
                    }))
                
        non_germ_entries = []
        non_germ_entries_for_data = []
        
        # Store UIDs that should be excluded (when we have too many seeds)
        uids_to_exclude = set()

        # For each Group-Video combination
        for (group, video), count in group_video_counts.items():
            # Get germinated seeds for this group and video
            germinated = germ_data[(germ_data['Group'] == group) & 
                                (germ_data['Video'] == video)]
            
            # Get all UIDs for this group and video from original data
            group_video_data = data[(data['Group'] == group) & (data['Video'] == video)]
            group_video_uids = set(group_video_data['UID'].unique())
            
            # Get current UIDs for this group and video in germination data
            current_uids = set(germinated['UID'])
            
            # Find missing UIDs for this group and video
            missing_uids = group_video_uids - current_uids
            
            if count > 0:  # If count is provided
                # Calculate how many missing UIDs we should add
                germinated_count = len(germinated)
                needed_count = count - germinated_count

                if needed_count < 0:
                    # We have MORE detected seeds than the expected count
                    # Sort by earliest germination time and keep only the first 'count' seeds
                    sorted_germinated = germinated.sort_values('GerminationTime')
                    keep_uids = set(sorted_germinated.head(count)['UID'].values)
                    
                    # Find UIDs to exclude
                    exclude_uids = set(germinated['UID']) - keep_uids
                    uids_to_exclude.update(exclude_uids)
                    continue  # Skip to next group-video combination
                
                if needed_count == 0:
                    continue  # Perfect match between detected and expected count

                # If we need more seeds than we have identified UIDs for
                missing_uids = list(missing_uids)[:needed_count]
                if len(missing_uids) < needed_count:
                    # Create additional entries with artificial UIDs
                    for i in range(needed_count - len(missing_uids)):
                        missing_uids.append(f"Group_{group}_video_{video}_NonGerm_{i}")

            # Add entries for missing UIDs
            for uid in missing_uids:
                non_germ_entries.append({
                    'Group': group,
                    'Video': video,
                    'UID': uid,
                    'GerminationTime': np.nan,
                    'SeedSize': np.nan
                })

                non_germ_entries_for_data.append({
                    'Group': group,
                    'Video': video,
                    'UID': uid,
                    'ElapsedHours': 0,
                    'Perim.': np.nan,
                    'Area': np.nan,
                    'SeedCount': np.nan,
                    'GrowthRate': np.nan,
                    'RateChange': np.nan,
                    'ShortTermChange': np.nan,
                    'LongTermChange': np.nan,
                    'TrendDirection': np.nan,
                    'Germinated': False
                })

        # Filter out excess UIDs from germination data
        germ_data = germ_data[~germ_data['UID'].isin(uids_to_exclude)]
        
        # Filter out excess UIDs from main data
        data = data[~data['UID'].isin(uids_to_exclude)]

        # Add all non-germinated entries at once
        if non_germ_entries:
            germ_data = pd.concat([germ_data, pd.DataFrame(non_germ_entries)], 
                                ignore_index=True)
        
        if non_germ_entries_for_data:
            data = pd.concat([data, pd.DataFrame(non_germ_entries_for_data)], 
                        ignore_index=True)
        
        # Store the seed counts for each group-video in the data
        self.group_video_seed_counts = group_video_counts
        
        # Save processed data
        self.data = data
        self.germination_data = germ_data

        germ_data.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_Summary.tsv'), 
                        sep='\t', index=False)
        data.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_ProcessedData.tsv'),
                    sep='\t', index=False)
        
    def _calculate_statistics(self):
        """Calculate summary statistics for germination data."""
        if self.germination_data is None:
            raise ValueError("No germination data available. Run process_data first.")
        
        # Calculate group statistics
        stats = self.germination_data.groupby('Group').agg({
            'GerminationTime': ['count', 'mean', 'std'],
            'SeedSize': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        stats.columns = [
            'Group',
            'Germinated_Seeds',
            'Mean_Germination_Time',
            'SD_Germination_Time',
            'Mean_Seed_Size',
            'SD_Seed_Size'
        ]
        
        # Calculate ungerminated seeds
        ungerminated = (self.germination_data[self.germination_data['GerminationTime'].isna()]
                       .groupby('Group')
                       .size()
                       .reset_index()
                       .rename(columns={0: 'Ungerminated_Seeds'}))
        
        # Merge statistics
        self.statistics = stats.merge(ungerminated, on='Group', how='left')
        self.statistics.to_csv(os.path.join(self.plot_dirs['data'], 'Germination_Statistics.tsv'), 
                             sep='\t', index=False)
        
        return self.statistics

    def _create_plots(self):
        """Generate all visualization plots."""
        # Original group plots
        for group_name, group_data in self.data.groupby('Group'):
            self._plot_germination_curve(group_name, group_data)
            
            # Add per-video plots for this group
            for video_name, video_data in group_data.groupby('Video'):
                self._plot_germination_curve_per_video(group_name, video_name, video_data)
        
        # Survival curves
        self._plot_survival_curves(pairwise=False)
        self._plot_survival_curves(pairwise=True)

    def _plot_germination_curve(self, group_name, group_data):
        """
        Create comprehensive germination curve plot using manual seed counts.
        """
        videos = group_data['Video'].unique()
    
        # Sum up the seed counts for all videos in this group
        total_seeds = sum(self.group_video_seed_counts.get((group_name, video), 0) 
                        for video in videos)
        
        # If there are no specified counts, fall back to counting UIDs
        if total_seeds == 0:
            total_seeds = len(group_data['UID'].unique())
            
        time_points = sorted(group_data['ElapsedHours'].unique())
        germination_counts = []
        raw_counts = []  # Add list for raw counts
        
        for t in time_points:
            germinated = len(group_data[
                (group_data['ElapsedHours'] <= t) & 
                (group_data['Germinated'])
            ]['UID'].unique())
            raw_counts.append(germinated)  # Store raw count
            germination_counts.append((germinated / total_seeds) * 100)
        
        try:
            # Fit Hill function
            popt, _ = curve_fit(
                self.hill_function, 
                time_points,
                germination_counts,
                p0=[0, max(germination_counts), 2, np.median(time_points)],
                bounds=([0, 0, 0, 0], [10, 100, 20, max(time_points)])
            )
            
            # Generate smooth curves
            x_smooth = np.linspace(0, max(time_points), 1000)
            y_smooth = self.hill_function(x_smooth, *popt)
            
            # Calculate metrics
            final_germ_percent = max(germination_counts)
            TMGR = self.find_TMGR(popt)
            final_germinated = len(group_data[group_data['Germinated']]['UID'].unique())
            
            # Calculate T50s using fine grid
            x_fine = np.linspace(0, max(time_points), 10000)
            y_fine = self.hill_function(x_fine, *popt)
            
            # T50 calculations remain the same...
            idx_50_total = np.argmin(np.abs(y_fine - 50))
            t50_total = x_fine[idx_50_total]
            
            if final_germ_percent > 0:
                target_germinated = final_germ_percent / 2
                idx_50_germinated = np.argmin(np.abs(y_fine - target_germinated))
                t50_germinated = x_fine[idx_50_germinated]
            else:
                t50_germinated = None
            
            # Calculate RoG curve
            rog = [self.germination_rate(x, *popt) for x in x_smooth]
            
            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(7, 4))
            
            # Primary y-axis (percentage)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Germination Percentage (%)')
            
            # Secondary y-axis (counts)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Seeds')
            
            # Add detailed information to the title with proper spacing
            fig.suptitle(f'Germination Curve - {group_name}\n'
             f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)',
             fontsize=12, y=1.05)
            
            # Reference lines (percentage)
            ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            
            # Main curves
            scatter1 = ax1.scatter(time_points, germination_counts, alpha=0.5, 
                            color='black', label='Observed Data')
            line1 = ax1.plot(x_smooth, y_smooth, '-', color='blue', 
                        label='FPHF Fit')
            line2 = ax1.plot(x_smooth, rog, '--', color='green', 
                        label='Rate of Germination')
            
            # Time markers
            vline1 = ax1.axvline(x=TMGR, color='red', linestyle='--', 
                            label=f'TMGR: {TMGR:.1f}h')
            vline2 = ax1.axvline(x=t50_total, color='black', linestyle='--', 
                            label=f'T50 Total: {t50_total:.1f}h')
            if t50_germinated:
                vline3 = ax1.axvline(x=t50_germinated, color='gray', linestyle='--', 
                                label=f'T50 Germinated: {t50_germinated:.1f}h')
            
            # Set limits
            ax1.set_ylim(0, 100)
            ax2.set_ylim(0, total_seeds)
            ax1.set_xlim(0, max(time_points))
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend for primary axis only
            ax1.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
            
            # Save plot
            plot_name = os.path.join(self.plot_dirs['germination'], 
                                f'GerminationCurve_{group_name}')
            plt.tight_layout()
            self._save_plot(plot_name)
                        
        except RuntimeError as e:
            print(f"Could not fit curve for group {group_name}: {str(e)}")

    def _plot_survival_curves(self, pairwise=False):
        """
        Create Kaplan-Meier survival curve plots.
        
        Args:
            pairwise: If True, creates separate plots for each pair of groups
        """
        # Prepare survival data
        data_surv = self.germination_data.copy()
        max_time = self.data['ElapsedHours'].max()
        
        data_surv['Germinated'] = ~data_surv['GerminationTime'].isna()
        data_surv['GerminationTime'] = data_surv['GerminationTime'].fillna(max_time)
        
        unique_groups = sorted(data_surv['Group'].unique())
        
        if not pairwise:
            # Plot all groups together
            plt.figure(figsize=(7, 4))
            
            for group in unique_groups:
                group_data = data_surv[data_surv['Group'] == group]
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['GerminationTime'], 
                       event_observed=group_data['Germinated'])
                
                # Plot germination curve (1 - survival)
                plt.step(kmf.survival_function_.index, 
                        1 - kmf.survival_function_, 
                        where='post', 
                        linewidth=2,
                        label=group)  # Use group name directly for legend
            
            plot_name = os.path.join(self.plot_dirs['survival'], 
                                   'KaplanMeier_AllGroups')
            self._format_survival_plot(max_time, 'Kaplan Meier - All Groups')
            self._save_plot(plot_name)
            
        else:
            # Create pairwise plots
            for i, group1 in enumerate(unique_groups):
                for j, group2 in enumerate(unique_groups[i+1:], i+1):
                    plt.figure(figsize=(7, 4))
                    
                    for group in [group1, group2]:
                        group_data = data_surv[data_surv['Group'] == group]
                        kmf = KaplanMeierFitter()
                        kmf.fit(group_data['GerminationTime'], 
                               event_observed=group_data['Germinated'])
                        
                        plt.step(kmf.survival_function_.index, 
                                1 - kmf.survival_function_, 
                                where='post', 
                                linewidth=2,
                                label=group)  # Use group name directly for legend
                    
                    plot_name = os.path.join(self.plot_dirs['survival'], 
                                           f'KaplanMeier_{group1}_vs_{group2}')
                    self._format_survival_plot(max_time, 
                                           f'Kaplan Meier - {group1} vs {group2}')
                    self._save_plot(plot_name)

    def _save_plot(self, name, formats=('.png', '.pdf')):
        """Save plot in multiple formats."""
        for fmt in formats:
            plt.savefig(name + fmt, bbox_inches='tight', dpi=300)
        plt.close()

    def _format_survival_plot(self, max_time, title):
        """Format Kaplan-Meier plot with standard elements."""
        plt.xlim(0, max_time)
        plt.ylim(0, 1)
        plt.xlabel('Time (hours)')
        plt.ylabel('Fraction of Germinated Seeds')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Format legend with better positioning and style
        plt.legend(title='Groups',
                  loc='center right',
                  frameon=True,
                  edgecolor='black',
                  fancybox=True,
                  shadow=True)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

    def _plot_germination_curve_per_video(self, group_name, video_name, video_data):
        """
        Create germination curve plot for a specific video.
        
        Args:
            group_name: Name of the treatment group
            video_name: Name/ID of the video
            video_data: DataFrame containing data for this video
        """
        # Get total seeds from stored seed counts
        total_seeds = self.group_video_seed_counts.get((group_name, video_name), 0)
        
        # If there's no specified count, fall back to counting UIDs
        if total_seeds == 0:
            total_seeds = len(video_data['UID'].unique())

        time_points = sorted(video_data['ElapsedHours'].unique())
        germination_counts = []
        
        for t in time_points:
            germinated = len(video_data[
                (video_data['ElapsedHours'] <= t) & 
                (video_data['Germinated'])
            ]['UID'].unique())
            germination_counts.append((germinated / total_seeds) * 100)
        
        try:
            # Fit Hill function
            popt, _ = curve_fit(
                self.hill_function, 
                time_points,
                germination_counts,
                p0=[0, max(germination_counts), 2, np.median(time_points)],
                bounds=([0, 0, 0, 0], [10, 100, 20, max(time_points)])
            )
            
            # Generate smooth curves
            x_smooth = np.linspace(0, max(time_points), 1000)
            y_smooth = self.hill_function(x_smooth, *popt)
            
            # Calculate metrics
            final_germ_percent = max(germination_counts)
            TMGR = self.find_TMGR(popt)
            final_germinated = len(video_data[video_data['Germinated']]['UID'].unique())
            

            # Create plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(7, 4))
            
            # Add detailed information to the title
            plt.suptitle(f'Germination Curve - {group_name} - Video {video_name}', fontsize=14)
            plt.title(f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)', 
                    fontsize=10)

            # Primary y-axis (percentage)
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Germination Percentage (%)')
            
            # Secondary y-axis (counts)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Seeds')
            
            # Add detailed information to the title
            plt.suptitle(f'Germination Curve - {group_name}', fontsize=14)
            plt.title(f'Seeds: {final_germinated} germinated out of {total_seeds} total ({final_germ_percent:.1f}%)',
                    fontsize=10)
            
            # Reference lines (percentage)
            ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            
            # Main curves
            scatter1 = ax1.scatter(time_points, germination_counts, alpha=0.5, 
                            color='black', label='Observed Data')
            line1 = ax1.plot(x_smooth, y_smooth, '-', color='blue', 
                        label='FPHF Fit')
            
            # Time markers
            vline1 = ax1.axvline(x=TMGR, color='red', linestyle='--', 
                            label=f'TMGR: {TMGR:.1f}h')
            
            # Set limits
            ax1.set_ylim(0, 100)
            ax2.set_ylim(0, total_seeds)
            ax1.set_xlim(0, max(time_points))
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Add legend for primary axis only
            ax1.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
            
            # Save plot
            plot_name = os.path.join(self.plot_dirs['germination_video'], 
                                f'GerminationCurve_{group_name}_Video_{video_name}')
            self._save_plot(plot_name)
            
        except RuntimeError as e:
            print(f"Could not fit curve for group {group_name} video {video_name}: {str(e)}")
