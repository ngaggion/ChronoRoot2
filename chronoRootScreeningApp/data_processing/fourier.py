import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def process_data(df, data_column='GrowthSpeed', sync_column='SyncHour'):
    """
    Process time series data by synchronizing and normalizing it.
    Args:
        df: DataFrame with time series data
        sync_column: Column name to synchronize to (default: 'SyncHour')
    Returns:
        DataFrame with synchronized and normalized time series data
    """
    df = df.sort_values(['UID', sync_column]).reset_index(drop=True)

    normalized_dfs = []
    ffts = []
    
    # Process each UID independently
    for uid in df['UID'].unique():
        # Get data for this UID
        uid_data = df[df['UID'] == uid].copy()
        
        # Fill NaN values in the data column
        uid_data[data_column] = uid_data[data_column].fillna(0)  # Replace NaNs with zeros
        
        # Normalize data
        uid_data[data_column] = (uid_data[data_column] - uid_data[data_column].mean()) / (uid_data[data_column].std() + 1e-10)
        
        # Apply filters
        medFilt_25 = lambda x: signal.medfilt(x, kernel_size=min(25, len(x) - (len(x) % 2) + 1))
        uid_data['Trend'] = medFilt_25(uid_data[data_column])
        uid_data['Detrend'] = uid_data[data_column] - uid_data['Trend']
        
        # Calculate Fourier transforms - ensure no NaNs are present
        normalized_fft = np.abs(np.fft.fft(np.nan_to_num(uid_data[data_column])))
        detrend_fft = np.abs(np.fft.fft(np.nan_to_num(uid_data['Detrend'])))
        freqs = np.fft.fftfreq(len(uid_data), d=1)
        
        # Create FFT DataFrame for this UID
        fft_df = pd.DataFrame({
            'UID': uid,
            'Group': uid_data['Group'].iloc[0],
            'Signal': uid_data[data_column],
            'DetrendSignal': uid_data['Detrend'],
            'Frequency': freqs,
            'NormalizedFFT': normalized_fft,
            'DetrendFFT': detrend_fft,
            'Time': uid_data[sync_column]
        })
        
        normalized_dfs.append(uid_data)
        ffts.append(fft_df)
    
    if not normalized_dfs:
        raise ValueError("No data remained after filtering")
    
    # Combine all processed data
    normalized_df = pd.concat(normalized_dfs, ignore_index=True)
    fft_df = pd.concat(ffts, ignore_index=True)
    
    return normalized_df, fft_df


def plot_fourier_summary(data, fft, savepath=None):
    """
    Create a comprehensive 2x2 plot grid showing original signal, detrended signal,
    and both FFT analyses.
    """
    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2)
    
    # Original Signal
    ax1 = fig.add_subplot(gs[0, 0])
    sns.lineplot(data=fft, x='Time', y='Signal', hue='Group', ax=ax1, errorbar='se')
    # Add day markers
    for t in fft['Time'].unique():
        if t % 24 == 0:
            ax1.axvline(t, color='gray', alpha=0.3, linestyle='--')
    ax1.set_title('Original Signal', fontsize=14)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Normalized Growth Rate')
    
    # Add second x-axis with days
    ax1_2 = ax1.twiny()
    ax1_2.set_xlim(ax1.get_xlim())
    days = np.arange(0, max(fft['Time']) // 24 + 1)
    ax1_2.set_xticks(days * 24)
    ax1_2.set_xticklabels([f'Day {d}' for d in days])
    
    # Detrended Signal
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=fft, x='Time', y='DetrendSignal', hue='Group', ax=ax2, errorbar='se')
    # Add day markers
    for t in fft['Time'].unique():
        if t % 24 == 0:
            ax2.axvline(t, color='gray', alpha=0.3, linestyle='--')
    ax2.set_title('Detrended Signal', fontsize=14)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Detrended Growth Rate')
    
    # Normalized FFT
    ax3 = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=fft, x='Frequency', y='NormalizedFFT', hue='Group', ax=ax3)
    ax3.set_xlim(0, 0.5)
    ax3.axvline(1/24, color='red', linestyle='--', label='24h period')
    ax3.axvline(1/12, color='black', linestyle='--', label='12h period')
    ax3.set_title('Normalized FFT', fontsize=14)
    ax3.set_xlabel('Frequency (1/hour)')
    ax3.set_ylabel('Amplitude')
    
    # Add frequency labels
    ax3.text(1/24, ax3.get_ylim()[1], '24h', rotation=90, va='top')
    ax3.text(1/12, ax3.get_ylim()[1], '12h', rotation=90, va='top')
    
    # Detrended FFT
    ax4 = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=fft, x='Frequency', y='DetrendFFT', hue='Group', ax=ax4)
    ax4.set_xlim(0, 0.5)
    ax4.axvline(1/24, color='red', linestyle='--', label='24h period')
    ax4.axvline(1/12, color='black', linestyle='--', label='12h period')
    ax4.set_title('Detrended FFT', fontsize=14)
    ax4.set_xlabel('Frequency (1/hour)')
    ax4.set_ylabel('Amplitude')
    
    # Add frequency labels
    ax4.text(1/24, ax4.get_ylim()[1], '24h', rotation=90, va='top')
    ax4.text(1/12, ax4.get_ylim()[1], '12h', rotation=90, va='top')
        
    plt.tight_layout()
    if savepath:
        plt.savefig(os.path.join(savepath, 'fourier_summary.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(savepath, 'fourier_summary.pdf'), bbox_inches='tight')
    
    return fig

def store_results(results_dir, normalized_df, fft_df):
    """
    Store results in the specified directory.
    
    Args:
        results_dir: Directory to store results in
        normalized_df: DataFrame with processed time series data
        fft_df: DataFrame with FFT results
        periodicity_df: DataFrame with periodicity analysis results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    normalized_df.to_csv(os.path.join(results_dir, 'normalized_data.tsv'), sep='\t', index=False)
    fft_df.to_csv(os.path.join(results_dir, 'fft_results.tsv'), sep='\t', index=False)
