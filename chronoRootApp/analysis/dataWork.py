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

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import re
import pandas as pd
import os
from scipy import signal
import json

def dataWork(conf, pfile, folder, N_exp = None):
    data = pd.read_csv(pfile)
    shape = data.shape
    N = shape[0]
    # Read info from the filename

    dates = []
    for i in range(0,N):
        name = data['FileName'][i]
        nums = re.findall(r'\d+', name)
        date = nums[0] + '-' + nums[1] + '-' + nums[2] + '-' + nums[3] + ':' + nums[4] 
        dates.append(date)

    data.insert(data.shape[1], 'Date', dates)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d-%H:%M')

    # Checks for missing values

    timeStep = conf['timeStep']
    timeStep = pd.Timedelta(minutes=timeStep)
    first = data['Date'][0]
    last = data['Date'][N-1]
    dates = pd.date_range(first, last, freq=timeStep)

    for date in dates:
        if not data['Date'].isin([date]).any():        
            data = pd.concat([data, pd.DataFrame({'Date': [date]})], ignore_index=True)
            #print('Missing date: ' + str(date))

    data = data.sort_values(by=['Date'])
    data = data.reset_index(drop=True)
    data = data.fillna(method='ffill')
    
    N = data.shape[0]

    # Trims or expands the data to the desired number of measurements
    
    if N_exp is not None:
        if N > N_exp:
            data = data.iloc[0:N_exp]
        else:
            # add rows with last data
            for i in range(N,N_exp):
                data = pd.concat([data, pd.DataFrame(data.iloc[N-1]).T], ignore_index=True)

    # If there is a repeated date, it is updated by a time step

    shape = data.shape
    N = shape[0]

    for i in range(1,N):
        if data['Date'][i] == data['Date'][i-1]:
            data.loc[i, 'Date'] = data['Date'][i] + timeStep
        if data['Date'][i] < data['Date'][i-1]:
            data.loc[i, 'Date'] = data['Date'][i-1] + timeStep
    
    # Reads the pixel size

    path = os.path.abspath(os.path.join(folder, 'metadata.json'))
    with open(path) as f:
        metadata = json.load(f)
    pixel_size = metadata['pixel_size']
    
    # Beggining of the data processing

    mainRoot = data['MainRootLength'].to_numpy().astype('float')
    lateralRoots = data['LateralRootsLength'].to_numpy().astype('float')
    numlateralRoots = data['NumberOfLateralRoots'].to_numpy().astype('float')

    # Used to remove spurious values at the beggining
    space = 32
    for t in range(space, N//2):
        if numlateralRoots[t-space] == 0 and numlateralRoots[t] == 0:
            lateralRoots[t-space:t] = 0
            numlateralRoots[t-space:t] = 0

    # Smooth
    mainRoot = signal.medfilt(mainRoot, 9) 
    lateralRoots = signal.medfilt(lateralRoots, 9) 
    numlateralRoots = signal.medfilt(numlateralRoots, 9)

    # Check that the values never decreases
    for i in range(1, len(mainRoot)):
        dif = mainRoot[i] < mainRoot[i-1]
        if dif:
            mainRoot[i] = mainRoot[i-1]
        
        dif = numlateralRoots[i] < numlateralRoots[i-1]
        if dif and numlateralRoots[i-1] > 1:
            numlateralRoots[i] = numlateralRoots[i-1]

        dif = lateralRoots[i] < lateralRoots[i-1]
        if dif and lateralRoots[i-1] > 10:
            lateralRoots[i] = lateralRoots[i-1]

    # Multiply by pixel size
    mainRoot_mm = mainRoot.copy() * pixel_size
    lateralRoots_mm = lateralRoots.copy() * pixel_size
    
    data['MainRootLength (mm)'] = mainRoot_mm
    data['LateralRootsLength (mm)'] = lateralRoots_mm
    data['NumberOfLateralRoots'] = numlateralRoots
    data['TotalLength (mm)'] = mainRoot_mm + lateralRoots_mm

    data['FileName'].to_csv(os.path.abspath(os.path.join(folder, 'FilesAfterPostprocessing.csv')), index = False)

    # Removes original columns
    try:
        data = data.drop(columns=['FileName', 'Frame', 'MainRootLength', 'LateralRootsLength', 'TotalLength'])
    except:
        # For retrocompatibility
        data = data.drop(columns=['FileName', 'MainRootLength', 'LateralRootsLength', 'TotalLength'])
        
    # Reorders columns
    data = data[['Date', 'MainRootLength (mm)', 'LateralRootsLength (mm)', 'TotalLength (mm)', 'NumberOfLateralRoots']]
    # creates an elapsed time column, in hours
    data['ElapsedTime (h)'] = ((data['Date'] - data['Date'][0]).dt.total_seconds() / 3600).round(2)
    # Creates a column that is True if the hour is 00:00, checking hours and minutes
    data['NewDay'] = (data['Date'].dt.hour == 0) & (data['Date'].dt.minute == 0)

    data.to_csv(os.path.abspath(os.path.join(folder, 'PostProcess_Original.csv')), index = False)

    # Downsamples the data to a 60 minutes timestep
    # Make a copy to prevent modifying the original
    original_data = data.copy()
    
    # Set the index and ensure it's named correctly
    data = data.set_index('Date')
    data.index.name = 'Date'  # Explicitly ensure index name is 'Date'
    
    # Fix: Ensure alignment with hour boundaries by using a reference timestamp
    reference_timestamp = data.index[0].floor('H')  # Round down to nearest hour
    
    # Fix: Use explicit origin parameter to ensure proper alignment
    hour_data = data.resample('60T', origin=reference_timestamp).mean()
    
    # Fix: Calculate expected number of rows after resampling
    if N_exp is not None:
        expected_hour_count = (N_exp + 3) // 4  # Ceiling division
        
        # If we got fewer rows than expected, add missing hours
        if len(hour_data) < expected_hour_count:
            # Calculate how many hours we're missing
            missing_hours = expected_hour_count - len(hour_data)
            
            # Get the last hour in the data
            last_hour = hour_data.index[-1]
            
            # Create new hours after the last one
            new_hours = pd.date_range(start=last_hour + pd.Timedelta(hours=1), periods=missing_hours, freq='60T')
            
            # Create a DataFrame with the new hours, filled with the last values
            new_data = pd.DataFrame(index=new_hours)
            for column in hour_data.columns:
                new_data[column] = hour_data[column].iloc[-1]
            
            # Append the new data
            hour_data = pd.concat([hour_data, new_data])
        
        elif len(hour_data) > expected_hour_count:
            # If we got more rows than expected, trim the extra hours
            hour_data = hour_data.iloc[:expected_hour_count]
    
    # Reset index and ensure the column is named correctly
    data = hour_data.reset_index()
    
    # Fix: Ensure 'Date' column exists - handle any naming issues
    if 'Date' not in data.columns and 'index' in data.columns:
        data = data.rename(columns={'index': 'Date'})
    
    data['NewDay'] = (data['Date'].dt.hour == 0) & (data['Date'].dt.minute == 0)

    # Re estimates the elapsed time
    data['ElapsedTime (h)'] = ((data['Date'] - data['Date'][0]).dt.total_seconds() / 3600).round(0)
    # Rounds the number of lateral roots
    data['NumberOfLateralRoots'] = data['NumberOfLateralRoots'].round(0)

    # Calculates gradients for the root lengths
    mainRootGrad = np.gradient(data['MainRootLength (mm)'].to_numpy(), edge_order = 2)
    lateralRootsGrad = np.gradient(data['LateralRootsLength (mm)'].to_numpy(), edge_order = 2)
    totalRootsGrad = np.gradient(data['TotalLength (mm)'].to_numpy(), edge_order = 2)
    
    # Main over total
    mainOverTotal = data['MainRootLength (mm)'].to_numpy() / data['TotalLength (mm)'].to_numpy() * 100
    where_nans = np.isnan(mainOverTotal)
    mainOverTotal[where_nans] = 100.0
    mainOverTotal = signal.medfilt(mainOverTotal, 5)

    # Lateral density
    lateralDensity = data['LateralRootsLength (mm)'].to_numpy() / data['MainRootLength (mm)'].to_numpy()
    where_nans = np.isnan(lateralDensity)
    lateralDensity[where_nans] = 0.0
    lateralDensity = signal.medfilt(lateralDensity, 5)

    # Discrete lateral density
    discreteLateralDensity = 10 * data['NumberOfLateralRoots'].to_numpy() / data['MainRootLength (mm)'].to_numpy()
    where_nans = np.isnan(discreteLateralDensity)
    discreteLateralDensity[where_nans] = 0.0
    discreteLateralDensity = signal.medfilt(discreteLateralDensity, 5)

    # Adds the new columns
    data['MainRootLengthGrad (mm/h)'] = mainRootGrad
    data['LateralRootsLengthGrad (mm/h)'] = lateralRootsGrad
    data['TotalLengthGrad (mm/h)'] = totalRootsGrad
    data['MainOverTotal (%)'] = mainOverTotal
    data['LateralDensity (mm/mm)'] = lateralDensity
    data['DiscreteLateralDensity (LR/cm)'] = discreteLateralDensity
    
    data.to_csv(os.path.abspath(os.path.join(folder, 'PostProcess_Hour.csv')), index = False)

    return
 
