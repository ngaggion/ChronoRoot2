#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import MonomialBasis
import seaborn as sns
from scipy.stats import norm, mannwhitneyu
import os
import argparse
import sys
plt.switch_backend('agg')

def parse_arguments():
    """Parse command line arguments for FPCA analysis."""
    parser = argparse.ArgumentParser(description='Perform FPCA analysis on temporal data.')
    
    # Required parameters
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the input CSV file containing temporal data')
    parser.add_argument('--xcol', type=str, required=True,
                        help='Column name for X axis (e.g., "ElapsedTime (h)")')
    parser.add_argument('--ycols', type=str, nargs='+', required=True,
                        help='Column names for Y axis to analyze (e.g., "MainRootLength (mm)")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for generated plots and statistics')
    
    # Optional parameters with defaults
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Apply inverse rank normalization to FPC values')
    parser.add_argument('--components', type=int, default=2,
                        help='Number of functional principal components to compute (default: 2)')
    parser.add_argument('--groupby', type=str, default='Experiment',
                        help='Column to group data by (default: "Experiment")')
    parser.add_argument('--id_col', type=str, default='Plant_id',
                        help='Column containing unique identifiers (default: "Plant_id")')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 20],
                        help='Figure size in inches (width, height) (default: 10 20)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures (default: 300)')
    
    return parser.parse_args()

def run_fpca_analysis(args):
    """Run FPCA analysis based on provided arguments."""
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # If ".csv" in the file path, read the file as a CSV
    if ".csv" in args.file:
        temporal_data_df = pd.read_csv(args.file)
    elif ".tsv" in args.file:
        temporal_data_df = pd.read_csv(args.file, sep='\t')
    else:
        print("Unsupported file format. Please provide a CSV or TSV file.")
        sys.exit(1)

    # Convert group column to string
    temporal_data_df[args.groupby] = temporal_data_df[args.groupby].astype('str')
    temporal_data_df = temporal_data_df.sort_values(by=args.groupby)
    
    # Create a unique identifier combining ID and group
    temporal_data_df[args.id_col] = (temporal_data_df[args.id_col].astype('str') + 
                                    " (" + temporal_data_df[args.groupby] + ")")
    
    # Get unique experiments/groups
    experiments = temporal_data_df[args.groupby].unique()
    
    # Create a dictionary to get group ID for each plant
    get_expid = lambda plant_id: temporal_data_df.set_index(args.id_col)[args.groupby].to_dict()[plant_id]
    
    plt.ioff()
    
    # Process each Y column
    for magnitude in args.ycols:
        print(f"Processing {magnitude}...")
        name = magnitude.split(" ")[0]
        
        # Create magnitude dictionary with pivoted data
        magnitude_data = temporal_data_df.pivot(
            columns=args.id_col, 
            values=magnitude, 
            index=args.xcol
        ).dropna()
        
        # Create figure
        plt.figure(figsize=tuple(args.figsize))
        
        # Plot 1: Line plot with error bars
        plt.subplot(5, 2, 1)
        sns.lineplot(
            x=args.xcol, 
            y=magnitude, 
            hue=args.groupby, 
            data=temporal_data_df, 
            errorbar='se', 
            palette="tab10"
        )
        plt.title(magnitude)
        
        # Perform FPCA
        fpca = FPCA(n_components=args.components, components_basis=MonomialBasis)
        fpc_values = fpca.fit_transform(FDataGrid(magnitude_data.transpose()))
        
        # Create DataFrame with FPC values
        fpc_df = pd.DataFrame(fpc_values).set_index(magnitude_data.columns)
        fpc_df.columns = [f"FPC{i}" for i in range(1, fpca.n_components+1)]
        fpc_df = fpc_df.reset_index()
        fpc_df[args.groupby] = fpc_df[args.id_col].apply(get_expid)
        
        # Inverse Rank Normalization if requested
        if args.normalize:
            for j in range(1, fpca.n_components+1):
                fpc_df[f'FPC{j}_IRN'] = norm.ppf(fpc_df[f'FPC{j}'].rank() / (len(fpc_df) + 1))
        
        fpc_df = fpc_df.sort_values(by=args.groupby)
        
        # Plot 2: Explained variance text
        ax = plt.subplot(5, 2, 2)
        plt.axis('off')
        for fpc1 in range(1, args.components + 1):
            plt.text(
                0.01, 
                1 - 0.10*fpc1, 
                f'Explained variance by PC{fpc1}: {fpca.explained_variance_ratio_[fpc1-1] * 100:.2f}%', 
                fontsize=12, 
                color='black'
            )
        plt.text(
            0.01, 
            1 - 0.10*(args.components+1), 
            f'Total explained variance: {sum(fpca.explained_variance_ratio_) * 100:.2f}%', 
            fontsize=12, 
            color='black'
        )
        
        # Write statistical results
        with open(os.path.join(args.output, f"{name}_stats.txt"), 'w') as f:
            f.write('Using Mann Whitney U test to compare different experiments\n')
            
            for fpc1 in range(1, args.components + 1):
                f.write(f'Stats for PC{fpc1}\n')
                
                for i in range(0, len(experiments)-1):
                    for j in range(i+1, len(experiments)):
                        exp1 = experiments[i]
                        exp2 = experiments[j]
                        p_value = mannwhitneyu(
                            x=fpc_df[f"FPC{fpc1}"][fpc_df[args.groupby] == exp1],
                            y=fpc_df[f"FPC{fpc1}"][fpc_df[args.groupby] == exp2],
                        )[1]
                        
                        # Compare the p-value with the significance level
                        if p_value < 0.05:
                            f.write(f'Experiments {experiments[i]} and {experiments[j]} are significantly different. P-value: {p_value}\n')
                        else:
                            f.write(f'Experiments {experiments[i]} and {experiments[j]} are not significantly different. P-value: {p_value}\n')
                
                f.write('\n')
                
                # Box plots for each FPC
                ax = plt.subplot(5, 2, 1 + fpc1 * 2)
                fpc_col = f"FPC{fpc1}_IRN" if args.normalize else f"FPC{fpc1}"
                
                sns.boxplot(
                    data=fpc_df, 
                    x=args.groupby, 
                    hue=args.groupby, 
                    y=fpc_col, 
                    ax=ax, 
                    palette="tab10"
                )
                ax.set_title(f'Box plot for PC {fpc1}')
                
                # Interpretation plot for each FPC
                ax = plt.subplot(5, 2, 1 + fpc1 * 2 + 1)
                
                N = 10
                quantiles = np.arange(N+1) / N
                z_quantiles = np.quantile(fpc_values, quantiles, axis=0)[1:-1]
                
                N = 8
                # Create a color palette
                palette = sns.color_palette("coolwarm", N+1)
                
                for i in range(z_quantiles.shape[0]):
                    z_value = z_quantiles[i, fpc1-1]
                    z = z_value * np.identity(args.components)[:, (fpc1-1)]
                    curve = fpca.inverse_transform(z)
                    curve = [x[0] for x in curve.data_matrix[0]]
                    
                    # Get color for the current quantile
                    color = palette[i]
                    
                    # Plot the curve with the corresponding color
                    ax.plot(curve, color=color, label=f'Q {quantiles[i]:.2f}')
                
                ax.set_title(f"Interpretation of PC{fpc1}")
                ax.set_ylabel(magnitude)
                ax.set_xlabel(f"Time ({args.xcol.split(' ')[-1].strip('()')})")
                
                # Create a legend with the quantiles
                handles = [plt.Line2D([0,1], [0,1], color=palette[i], lw=2) for i in range(N+1)]
                labels = [f'{z_quantiles[i, fpc1-1]:.2f}' for i in range(N+1)]
                ax.legend(
                    handles, 
                    labels, 
                    title=f'FPC{fpc1} Value', 
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left'
                )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"{name}.png"), dpi=args.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(args.output, f"{name}.svg"), dpi=args.dpi, bbox_inches='tight')
        plt.close()
        plt.cla()
        plt.clf()
        
        print(f"Saved plots for {magnitude} to {args.output}")

if __name__ == "__main__":
    args = parse_arguments()
    run_fpca_analysis(args)