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
import cv2
import graph_tool.all as gt
import json

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


def plot_individual_plant(savepath, dataframe, name):
    plt.ioff()
    
    fig, ax = plt.subplots(figsize = (9, 6), dpi = 300)

    dataframe.plot(x = 'ElapsedTime (h)', y = 'MainRootLength (mm)', ax = ax, color = 'g')
    dataframe.plot(x = 'ElapsedTime (h)', y = 'LateralRootsLength (mm)', ax = ax, color = 'b')
    ax.set_title('%s' % name, pad=20)
    ax.set_ylabel('Length (mm)')

    # increase font sizes 
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.title.set_size(18)
    ax.legend(fontsize = 18)

    # Create a second x-axis for displaying the hours
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Calculate the total number of days
    total_days = np.ceil(dataframe['ElapsedTime (h)'].max() / 24).astype(int)

    # Create day ticks
    day_ticks = np.arange(24, total_days * 24 + 1, 24)

    # Set day ticks and labels
    ax2.set_xticks(day_ticks)
    ax2.set_xticklabels([f'Day {i}' for i in range(1, total_days+1)], rotation=45)

    # Customize the appearance of the ticks and tick labels
    ax2.tick_params(axis='x', which='major', length=8, width=2, color='black')
    ax2.tick_params(axis='x', which='minor', length=4, width=1, color='black')
    ax2.tick_params(axis='x', which='major', labelsize=12)
    
    ax.figure.savefig(os.path.join(savepath, name), dpi = 300, bbox_inches = 'tight')
        
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
                subdata = subdata.groupby(['Experiment', 'Plant_id']).mean().reset_index()
    
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
        subdata = subdata.groupby(['Experiment', 'Plant_id']).mean().reset_index()
        subdata = subdata.groupby(['Experiment']).agg({'MainRootLength (mm)': ['count', 'mean', 'std'],
                                                      'LateralRootsLength (mm)': ['mean', 'std'], 
                                                      'TotalLength (mm)': ['mean', 'std'], 
                                                      'NumberOfLateralRoots': ['mean', 'std'], 
                                                      'DiscreteLateralDensity (LR/cm)': ['mean', 'std'], 
                                                      'MainOverTotal (%)': ['mean', 'std']})
        
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

    fig3 = plt.figure(figsize=(18,12), constrained_layout=True)
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

    f_ax1.annotate('(A)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax2.annotate('(B)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax3.annotate('(C)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax4.annotate('(D)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax5.annotate('(E)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')
    f_ax6.annotate('(F)',(0.47,-0.15), xycoords="axes fraction", fontsize=15, weight = 'bold')

    plt.savefig(os.path.join(savepath,'Temporal_Subplots_Mean_SE.svg'),dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(savepath,'Temporal_Subplots_Mean_SE.png'),dpi=300, bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close('all')

def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass
    return

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_atlases(psave, days = [6,7,8,9,10], rotateRoot = True):
    center = [500, 800]
    
    atlas = np.zeros([2500, 1600], dtype = 'uint8')
    aux = np.zeros([2500, 1600], dtype = 'uint8')
    atlas2 = np.zeros([2500, 1600, 3], dtype = 'uint8')
    atlasroot = np.zeros([2500, 1600], dtype = 'float64')

    paths = load_path(psave,'*/*/*/Results*')

    dias = days
    rotate =  rotateRoot
    
    plt.ioff()
    frames = []
    
    set_of_atlases = []

    for dia in dias:
        dia = int(dia)
        atlas2[:,:,:] = 255
        atlas[:,:] = 0
        aux[:,:] = 0
        atlasroot[:,:] = 0
        
        area_bbox_list = []
        area_chull_list = []
        aspect_ratio_list = []
        densidad_lateral_list = []
        densidad_total_list = []
        densidad_lateral_bbox_list = []
        densidad_total_bbox_list = []       
        width_list = []
        height_list = []
        
        for i in paths:
            data = load_path(i, 'PostProcess_Hour.csv')
            data = pd.read_csv(data[0])
            
            gpath = os.path.join(i, 'Graphs/')        
            graphs = load_path(gpath, "*.xml.gz")

            spath = os.path.join(i, 'Images/Seg/')
            
            segmentaciones = load_path(spath, "*.png")
            
            index = dia * 24 * 4
            if index >= len(segmentaciones):
                index = -1

            seg = segmentaciones[index]
            
            img = cv2.imread(seg, 0)
            shape = img.shape
                        
            graph_path = seg.replace(spath,gpath).replace('png','xml.gz')
            if index != -1:
                try:
                    g = gt.load_graph(graph_path)
                except:
                    g = gt.load_graph(graphs[-1])
            else:
                g = gt.load_graph(graphs[index])
                
            path = os.path.abspath(os.path.join(i, 'metadata.json'))
            with open(path) as f:
                metadata = json.load(f)

            seed = metadata['seed']

            img = np.pad(img,((0,0),(200,0)))
            shape = img.shape

            for j in g.get_vertices():
                tipo = g.vp.nodetype[j]
                if tipo == 'FTip':
                    end1 = np.array(g.vp.pos[j])
                if tipo == 'Ini':
                    end2 = np.array(g.vp.pos[j])

            # choose the root end, because the endpoints may be inverted
            if end1[1] > end2[1]:
                end = end1
            else:
                end = end2
            
            # to add the padding
            seed[0] = seed[0] + 200
            end[0] = end[0] + 200
            
            if rotate:
                v1 = np.array([0, 1])
                v2 = unit_vector(end - seed)   
                angle_rad = angle_between(v1, v2)
                
                if v2[0] > 0:
                    angle = -np.rad2deg(angle_rad)
                else:
                    angle = np.rad2deg(angle_rad)
                
                rot_mat = cv2.getRotationMatrix2D(tuple([seed[0],seed[1]]), angle, 1.0)
                result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
                
                _, result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)
            else:
                result = img
                
            y0 = int(center[0]-seed[1])
            if y0 < 0:
                print('CHECK CENTER')
                
            y1 = int(y0 + shape[0])
            
            x0 = int(center[1]-seed[0])
            x1 = int(x0 + shape[1])
                    
            # Finding contours for the thresholded image
            contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
                
            # create hull array for convex hull points
            hull = []
            
            # calculate points for each contour
            for j in range(len(contours)):
                # creating convex hull object for each contour
                hull.append(cv2.convexHull(contours[j], False))
            
            aux[:,:] = 0
            # draw contours and hull points
            for j in range(len(contours)):
                cv2.drawContours(aux[y0:y1, x0:x1], hull, j, 1, -1)
                cv2.drawContours(atlas2[y0:y1, x0:x1], hull, j, (255, 0, 0), 3)
            
            atlasroot[y0:y1,x0:x1] += result/255
            atlas += aux
            
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            dist = cv2.pointPolygonTest(biggest_contour,(seed[0], seed[1]), True)
            is_in = cv2.pointPolygonTest(biggest_contour,(seed[0], seed[1]), False) > 0
            
            if is_in or dist < 30:
                
                # create hull array for convex hull points
                hull = []
                
                # calculate points for each contour
                hull.append(cv2.convexHull(biggest_contour, False))
                
                x,y,w,h = cv2.boundingRect(hull[0])
                # cv2.rectangle(result,(x,y),(x+w,y+h),125,1)
                
                pixel_size = 0.04
                
                area_bbox = w * h * (pixel_size * pixel_size)
                area_chull = cv2.contourArea(hull[0]) * (pixel_size * pixel_size)
                aspect_ratio = h/w
                
                indice = int(index//4)

                if indice >= len(data['TotalLength (mm)']) or index == -1:
                    indice = len(data['TotalLength (mm)']) - 1
                    
                try:
                    totalRoot = data['TotalLength (mm)'][indice]
                    lateralRoots = data['LateralRootsLength (mm)'][indice]
                    densidad_lateral = lateralRoots/area_chull
                    densidad_total = totalRoot/area_chull

                    densidad_lateral_bbox = lateralRoots/area_bbox
                    densidad_total_bbox = totalRoot/area_bbox

                    area_bbox_list.append(area_bbox)
                    area_chull_list.append(area_chull) 
                    aspect_ratio_list.append(aspect_ratio)
                    densidad_lateral_list.append(densidad_lateral)
                    densidad_total_list.append(densidad_total)
                    densidad_lateral_bbox_list.append(densidad_lateral_bbox)
                    densidad_total_bbox_list.append(densidad_total_bbox)
                    width_list.append(w * pixel_size)
                    height_list.append(h * pixel_size)
                except:
                    print("LEN",len(data['TotalLength (mm)']))
                    print(indice)
                
            else:
                print('No contour')
    
        listas = list(zip(aspect_ratio_list, densidad_lateral_list, densidad_lateral_bbox_list, 
                          densidad_total_list, densidad_total_bbox_list, area_chull_list, 
                          area_bbox_list, width_list, height_list))
        dataframe = pd.DataFrame(listas, columns =['Convex Hull Aspect Ratio','Lateral Root Area Density',
                                                   'Lateral Root Area Density BBOX','Total Root Area Density', 
                                                   'Total Root Area Density BBOX', 'Convex Hull Area', 
                                                   'Bounding Box Area', 'Convex Hull Width', 'Convex Hull Height'])
        dataframe['Day'] = dia
        frames.append(dataframe)

        set_of_atlases.append([atlas.copy(), atlas2.copy(), atlasroot.copy()])
    
    frames = pd.concat(frames, ignore_index = True)

    return set_of_atlases, frames

def plot_atlases(atlas, atlas2, atlasroot, savepath, name, day = None):
    plt.ioff()

    plt.figure(figsize=(9, 4))

    plt.subplot(1,3,1)
    plt.imshow(atlasroot, cmap='jet', vmin = 0, vmax = 25)
    plt.title("Accumulated roots")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(atlas2, cmap='jet', vmin = 0, vmax = 25)
    plt.title("Convex hull contours")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(atlas, alpha = 0.6)
    plt.title("Accumulated convex hulls")
    plt.axis('off')

    if day is not None:
        name = name + ' - Day: ' + str(day)
    else:
        name = name + ' - Last Day'

    plt.suptitle(name)
    
    os.makedirs(os.path.join(savepath, "Per Experiment"), exist_ok = True)
    plt.savefig(os.path.join(savepath, "Per Experiment", name), dpi = 300, bbox_inches = 'tight')

    plt.cla()
    plt.clf()
    plt.close('all')


def plot_combined_atlases(folder):
    images_per_day = {}
    
    separated = os.path.join(folder, 'Per Experiment')

    for filename in os.listdir(separated):
        name = filename.split('.')[0]
        day = int(name.split(' ')[-1])

        if day not in images_per_day:
            images_per_day[day] = [filename]
        else:
            images_per_day[day].append(filename)
    
    for day in images_per_day:
        images = images_per_day[day]
        images.sort(key=natural_key)

        fig, axs = plt.subplots(len(images), 1, figsize=(9, 4*len(images)))

        for i, image in enumerate(images):
            image_path = os.path.join(separated, image)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title('')

        # remove all padding
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(os.path.join(folder, 'Qualitative - Day ' + str(day)), dpi = 300, bbox_inches = 'tight')
        plt.cla()
        plt.clf()
        plt.close('all')

    return
        

def plot_convex_hull(savepath, frame, name = ''):
    plt.ioff()

    n_types = len(frame['Experiment'].unique())

    fig, ax = plt.subplots()

    sns.violinplot(x = 'Day', y = 'Convex Hull Area', data=frame, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False)
    ax = sns.swarmplot(x = 'Day', y = 'Convex Hull Area', data=frame, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Convex Hull Area')
    ax.set_ylabel('Area (mm²)')
    
    plt.savefig(os.path.join(savepath, "Convex Hull Area.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Convex Hull Area.svg"), dpi = 300, bbox_inches = 'tight')

    fig, ax = plt.subplots()
    
    # remove plants without lateral roots
    frame2 = frame[frame['Lateral Root Area Density'] > 0].reset_index(drop = True)
    hue_order = frame['Experiment'].unique()

    sns.violinplot(x = 'Day', y = 'Lateral Root Area Density', data=frame2, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False, hue_order = hue_order)
    ax = sns.swarmplot(x = 'Day', y = 'Lateral Root Area Density', data=frame2, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2, 
                hue_order = hue_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Lateral Roots Area Density')
    ax.set_ylabel('LR / convex hull area (mm/mm²)')

    plt.savefig(os.path.join(savepath, "Lateral Root Area Density.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Lateral Root Area Density.svg"), dpi = 300, bbox_inches = 'tight')

    fig, ax = plt.subplots()
    
    sns.violinplot(x = 'Day', y = 'Convex Hull Aspect Ratio', data=frame, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False)
    ax = sns.swarmplot(x = 'Day', y = 'Convex Hull Aspect Ratio', data=frame, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Aspect Ratio')
    ax.set_ylabel('Aspect Ratio (height/width)')

    plt.savefig(os.path.join(savepath, "Convex Hull Aspect Ratio.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Convex Hull Aspect Ratio.svg"), dpi = 300, bbox_inches = 'tight')

    fig, ax = plt.subplots()
    
    # remove plants without lateral roots
    frame2 = frame[frame['Total Root Area Density'] > 0].reset_index(drop = True)
    hue_order = frame['Experiment'].unique()

    sns.violinplot(x = 'Day', y = 'Total Root Area Density', data=frame2, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False, hue_order = hue_order)
    ax = sns.swarmplot(x = 'Day', y = 'Total Root Area Density', data=frame2, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2, 
                hue_order = hue_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Total Root Area Density')
    ax.set_ylabel('TR / convex hull area (mm/mm²)')

    plt.savefig(os.path.join(savepath, "Total Root Area Density.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Total Root Area Density.svg"), dpi = 300, bbox_inches = 'tight')

    fig, ax = plt.subplots()
    
    hue_order = frame['Experiment'].unique()

    sns.violinplot(x = 'Day', y = 'Convex Hull Width', data=frame2, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False, hue_order = hue_order)
    ax = sns.swarmplot(x = 'Day', y = 'Convex Hull Width', data=frame2, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2, 
                hue_order = hue_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Convex Hull Width')
    ax.set_ylabel('Width (mm)')

    plt.savefig(os.path.join(savepath, "Convex Hull Width.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Convex Hull Width.svg"), dpi = 300, bbox_inches = 'tight')

    fig, ax = plt.subplots()
    
    hue_order = frame['Experiment'].unique()

    sns.violinplot(x = 'Day', y = 'Convex Hull Height', data=frame2, hue = 'Experiment', inner=None, 
                    showmeans=True, zorder=2, legend = False, hue_order = hue_order)
    ax = sns.swarmplot(x = 'Day', y = 'Convex Hull Height', data=frame2, hue = 'Experiment', dodge= True, 
                size = 4, palette = 'muted', edgecolor='black', linewidth = 0.5, zorder=1, s = 2, 
                hue_order = hue_order)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:n_types], labels[0:n_types], loc=2)

    ax.set_title('Convex Hull Height')
    ax.set_ylabel('Height (mm)')

    plt.savefig(os.path.join(savepath, "Convex Hull Height.png"), dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(savepath, "Convex Hull Height.svg"), dpi = 300, bbox_inches = 'tight')
    
    plt.cla()
    plt.clf()
    plt.close('all')

    # Group data by Day and Experiment, then calculate mean and standard deviation for each metric
    summary_data = frame.groupby(['Day', 'Experiment']).agg({'Convex Hull Area': ['count', 'mean', 'std'],
                                                            'Lateral Root Area Density': ['mean', 'std'],
                                                            'Convex Hull Aspect Ratio': ['mean', 'std'],
                                                            'Total Root Area Density': ['mean', 'std'],
                                                            'Convex Hull Width': ['mean', 'std'],
                                                            'Convex Hull Height': ['mean', 'std']})

    # Flatten the multi-index columns for better readability
    summary_data.columns = [' '.join(col).strip() for col in summary_data.columns.values]

    # Reset index to make Day and Experiment as columns
    summary_data = summary_data.reset_index()

    # Rename columns for better readability
    summary_data.columns = ['Day', 'Experiment', 'N Samples', 
                            'Convex Hull Area Mean', 'Convex Hull Area Std', 
                            'Lateral Root Area Density Mean', 'Lateral Root Area Density Std',
                            'Convex Hull Aspect Ratio Mean', 'Convex Hull Aspect Ratio Std',
                            'Total Root Area Density Mean', 'Total Root Area Density Std',
                            'Convex Hull Width Mean', 'Convex Hull Width Std',
                            'Convex Hull Height Mean', 'Convex Hull Height Std']
    
    summary_data = summary_data.round(3)

    # Save summary data as a table
    summary_data.to_csv(os.path.join(savepath, "Summary Table.csv"), index=False)

    
def performStatisticalAnalysisConvexHull(conf, data, metric):
    data['Experiment'] = data['Experiment'].astype(str)
    data['Day'] = data['Day'].astype(str)
    
    UniqueExperiments = data['Experiment'].unique()
    N_exp = int(len(UniqueExperiments))
    
    days = conf['daysConvexHull'].split(',')
    
    # Create a text file to store the results
    reportPath = os.path.join(conf['MainFolder'],'Report', 'Convex Hull and Area Analysis')
    
    if "/" in metric:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric.replace('/',' over '))
    else:
        reportPath_stats = os.path.join(reportPath, '%s Stats.txt' % metric)
        
    # First row should say "Using Mann Whitney U test to compare the growth speed of different experiments"
    with open(reportPath_stats, 'w') as f:
        f.write('Using Mann Whitney U test to compare different experiments\n')
        
        if metric == 'Lateral Root Area Density':
            f.write('For Lateral Root Area Density, it removes all plants without lateral roots\n\n')
         
        for day in days:                       
            # Compare every pair of experiments with Mann-Whitney U test
            f.write('Day: ' + str(day) + '\n')
            subdata = data[data['Day'] == day]
            
            if metric == 'Lateral Root Area Density':
                subdata = subdata[subdata['Lateral Root Area Density'] > 0].reset_index(drop = True)
            
            for i in range(0, N_exp-1):
                for j in range(i+1, N_exp):
                    exp1 = subdata[subdata['Experiment'] == UniqueExperiments[i]][metric]
                    exp2 = subdata[subdata['Experiment'] == UniqueExperiments[j]][metric]
                    
                    # Perform Mann-Whitney U test
                    try:
                        if len(exp1) == 0 or len(exp2) == 0:
                            raise Exception()
                            
                        U, p = stats.mannwhitneyu(exp1, exp2)
                        p = round(p, 6)
                        
                        # Write the mean value of each experiment
                        f.write('Mean ' + UniqueExperiments[i] + ': ' + str(round(exp1.mean(), 2)) + '\n')
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