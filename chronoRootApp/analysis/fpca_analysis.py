import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda import FDataGrid, FData
from skfda.preprocessing.dim_reduction import FPCA

from skfda.representation.basis import (
    MonomialBasis,
)

import seaborn as sns
from scipy.stats import norm, mannwhitneyu
import os
import json
import sys
plt.switch_backend('agg')

if __name__ == "__main__":
    conf_path = sys.argv[1]

    COLUMNS = [
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'NumberOfLateralRoots',
    'MainOverTotal (%)',
    'DiscreteLateralDensity (LR/cm)'
    ]

    with open(conf_path, 'r') as file:
            conf = json.load(file)

    path = conf['MainFolder']
    report = os.path.join(path, "Report", "FPCA")
    os.makedirs(report, exist_ok=True)

    basis = MonomialBasis

    inverse_rank_normalize = conf['normFPCA']
    number_of_components = int(conf['numComponentsFPCAField'])

    temporal_data_df = pd.read_csv(os.path.join(os.path.join(path, "Report", "Temporal_Data.csv")))
    temporal_data_df['Experiment'] = temporal_data_df['Experiment'].astype('str')
    temporal_data_df = temporal_data_df.sort_values(by="Experiment")

    experiments = temporal_data_df.Experiment.unique()

    magnitudes_dict = { magnitude: temporal_data_df.pivot(columns="Plant_id", values=magnitude, index="ElapsedTime (h)").dropna() for magnitude in COLUMNS}    
    get_expid = lambda plant_id: temporal_data_df.set_index("Plant_id")["Experiment"].to_dict()[plant_id]

    plt.ioff()

    for magnitude in COLUMNS:
        name = magnitude.split(" ")[0]
        
        plt.figure(figsize = (10, 20))

        plt.subplot(5,2,1)

        sns.lineplot(x='ElapsedTime (h)', y=magnitude, hue='Experiment', data = temporal_data_df, errorbar='se', palette="tab10")
        plt.title(magnitude)

        fpca = FPCA(n_components=number_of_components, components_basis=basis)
        fpc_values = fpca.fit_transform(FDataGrid(magnitudes_dict[magnitude].transpose()))
        
        fpc_df = pd.DataFrame(fpc_values).set_index(magnitudes_dict[magnitude].columns)
        fpc_df.columns = [ f"FPC{i}" for i in range(1, fpca.n_components+1) ]
        fpc_df = fpc_df.reset_index()
        fpc_df["Experiment"] = fpc_df.Plant_id.apply(get_expid)
        
        # Inverse Rank Normalization for Standard Normal Distribution
        for j in range(1, fpca.n_components+1):
            fpc_df[f'FPC{j}_IRN'] = norm.ppf(fpc_df[f'FPC{j}'].rank() / (len(fpc_df) + 1))
        fpc_df = fpc_df.sort_values(by="Experiment")
        
        ax = plt.subplot(5, 2, 2)
        plt.axis('off')
        for fpc1 in range(1, number_of_components + 1): 
            plt.text(0.01, 1 - 0.10*fpc1, 'Explained variance by PC%s: ' %fpc1+ '{:.2f}%'.format(fpca.explained_variance_ratio_[fpc1-1] * 100), fontsize=12, color='black')
        plt.text(0.01, 1 - 0.10*(number_of_components+1), 'Total explained variance: ' + '{:.2f}%'.format(sum(fpca.explained_variance_ratio_) * 100), fontsize=12, color='black')

        with open(os.path.join(report, name + "_stats.txt"), 'w') as f:
            f.write('Using Mann Whitney U test to compare different experiments\n')
            
            for fpc1 in range(1, number_of_components + 1):        
                f.write('Stats for PC%s\n'%fpc1)
            
                for i in range(0, len(experiments)-1):
                    for j in range(i+1, len(experiments)):
                        exp1 = experiments[i]
                        exp2 = experiments[j]
                        p_value = mannwhitneyu(
                            x = fpc_df[f"FPC{fpc1}"][fpc_df.Experiment == exp1], 
                            y = fpc_df[f"FPC{fpc1}"][fpc_df.Experiment == exp2],                         
                        )[1]

                        # Compare the p-value with the significance level
                        if p_value < 0.05:
                            f.write('Experiments ' + str(experiments[i]) + ' and ' + str(experiments[j]) + ' are significantly different. P-value: ' + str(p_value) + '\n')
                        else:
                            f.write('Experiments ' + str(experiments[i]) + ' and ' + str(experiments[j]) + ' are not significantly different. P-value: ' + str(p_value) + '\n')           
                
                f.write('\n')

                ax = plt.subplot(5,2, 1 + fpc1 * 2)

                sns.boxplot(data=fpc_df, x = "Experiment", hue="Experiment", y=f"FPC{fpc1}{'_IRN' if inverse_rank_normalize else ''}", ax=ax, palette="tab10");
                ax.set_title('Box plot for PC %s' %fpc1)
                
                ax = plt.subplot(5,2, 1 + fpc1 * 2 + 1)

                N = 10
                quantiles = np.arange(N+1) / N
                z_quantiles = np.quantile(fpc_values, quantiles, axis=0)
                # Create a color palette
                palette = sns.color_palette("coolwarm", N+1)

                for i in range(z_quantiles.shape[0]):
                    z_value = z_quantiles[i, fpc1-1]
                    z = z_value * np.identity(number_of_components)[:, (fpc1-1)]
                    curve = fpca.inverse_transform(z)
                    curve = [x[0] for x in curve.data_matrix[0]]
                    
                    # Get color for the current quantile
                    color = palette[i]
                    
                    # Plot the curve with the corresponding color
                    ax.plot(curve, color=color, label=f'Q {quantiles[i]:.2f}')

                ax.set_title(f"Interpretation of PC{fpc1}")

                # Create a legend with the quantiles
                handles = [plt.Line2D([0,1], [0,1], color=palette[i], lw=2) for i in range(N+1)]
                labels = [f'{z_quantiles[i, fpc1-1]:.2f}' for i in range(N+1)]
                ax.legend(handles, labels, title=f'FPC{fpc1} Value', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(report, name + ".png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(report, name + ".svg"), dpi=300, bbox_inches='tight')
            plt.close()
            plt.cla()
            plt.clf()
    