## firstly install package actipy (e.g. in a Notebook write !pip install actipy)

import actipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# UPLOAD
data, info = actipy.read_device("FILE.cwa",  ## insert file name
                                 calibrate_gravity=False,
                                 detect_nonwear=True)

# RENAME AND CONVERT
data = data.rename(columns={'x': 'acc_y', 'y': 'acc_x', 'z': 'acc_z'})
data = data.rename(columns={'gyro_x': 'gyro_y_temp', 'gyro_y': 'gyro_x_temp'})
data = data.rename(columns={'gyro_x_temp': 'gyro_x', 'gyro_y_temp': 'gyro_y'})

data['acc_y'] = (data['acc_y'] - 1) * 9.80665
data['acc_x'] = data['acc_x'] * 9.80665
data['acc_z'] = data['acc_z'] * 9.80665

data = data.astype(np.float64)
data['time'] = (data.index.astype(np.int64) / 10**6).astype(np.int64)
data['time'] = data['time'] - data['time'].min()

# CUSTOMIZED WINDOW
start_time_ms = 1000 ##ms, adjust as needed 
end_time_ms = 10000 ##ms, adjust as needed

def crop_data(df, start, end):
    return df[(df['time'] >= start) & (df['time'] <= end)].copy()

data_cropped = crop_data(data, start_time_ms, end_time_ms)

data_cropped['time'] = data_cropped['time'] - data_cropped['time'].min()

# VISUALIZATION
plt.figure(figsize=(15, 6))
plt.plot(data_cropped['time'], data_cropped['acc_x'], label='Acc X', alpha=0.7)
plt.plot(data_cropped['time'], data_cropped['acc_y'], label='Acc Y', alpha=0.7)
plt.plot(data_cropped['time'], data_cropped['acc_z'], label='Acc Z', alpha=0.7)
plt.title(f'Accelerazioni Ritagliate e Resettate (Originale: {start_time_ms}ms - {end_time_ms}ms)')
plt.xlabel('Tempo (ms dalla selezione)')
plt.ylabel('Accelerazione (m/s²)')
plt.legend()
plt.grid(True)
plt.show()

# EXPORT
columns_to_export = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'time']
df_export = data_cropped[columns_to_export]
csv_filename = 'dati_elaborati.csv'
df_export.to_csv(csv_filename, index=False)

print(f"File '{csv_filename}' creato con {len(df_export)} righe (il tempo ora parte da 0).")
display(df_export.head())
