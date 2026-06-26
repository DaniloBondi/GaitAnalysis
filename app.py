from shiny import App, render, ui, reactive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    welch,
    butter,
    filtfilt,
    savgol_filter,
    find_peaks
)

from scipy.fft import fft, fftfreq
from sklearn.metrics import mutual_info_score

import nolds
import warnings

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='sklearn.metrics'
)

app_ui = ui.page_sidebar(

    ui.sidebar(

        ui.input_file(
            "file",
            "Upload CSV File",
            accept=[".csv"],
            multiple=False
        ),

        ui.input_numeric(
            "start_time",
            "Start Time (ms)",
            value=500,
            min=0
        ),

        ui.input_numeric(
            "end_time",
            "End Time Before End (ms)",
            value=500,
            min=0
        ),

        ui.input_numeric(
            "fs",
            "Sampling Frequency (Hz)",
            value=400,
            min=1
        ),

        ui.input_numeric(
            "cutoff",
            "Low-pass Filter Cutoff (Hz)",
            value=20,
            min=1
        ),

        ui.input_numeric(
            "height",
            "Subject Height (m)",
            value=1.70,
            min=1.0,
            max=2.3,
            step=0.01
        ),

        ui.input_action_button(
            "analyze",
            "Run Analysis",
            class_="btn-primary"
        ),

        width=300
    ),

    ui.navset_card_tab(

        ui.nav_panel(
            "Raw Data",
            ui.output_plot("raw_plot")
        ),

        ui.nav_panel(
            "Step Detection",
            ui.output_plot("step_plot")
        ),

        ui.nav_panel(
            "Filtered vs Smoothed",
            ui.output_plot("filtered_smoothed_plot")
        ),

        ui.nav_panel(
            "Step Metrics",
            ui.output_text_verbatim("step_metrics")
        ),

        ui.nav_panel(
            "Spatiotemporal Metrics",
            ui.output_text_verbatim("spatiotemporal_metrics")
        ),

        ui.nav_panel(
            "Gait Metrics X",
            ui.output_text_verbatim("gait_metrics_x")
        ),

        ui.nav_panel(
            "Gait Metrics Y",
            ui.output_text_verbatim("gait_metrics_y")
        ),

        ui.nav_panel(
            "Gait Metrics Z",
            ui.output_text_verbatim("gait_metrics_z")
        ),

        ui.nav_panel(
            "Lyapunov Exponents",
            ui.output_text_verbatim("lyapunov_results")
        )
    )
)

def server(input, output, session):

    analysis_results = reactive.value(None)


    @reactive.effect
    @reactive.event(input.analyze)
    def _():

        file_info = input.file()

        if file_info is None:

            ui.notification_show(
                "Please upload a CSV file first",
                type="warning"
            )

            return

        try:


            df = pd.read_csv(
                file_info[0]["datapath"]
            )

            acc_z = -df['accZ']
            acc_x = df['accX']
            acc_y = df['accY'] - 9.81

            gyro_z = df['gyroZ']
            gyro_x = df['gyroX']
            gyro_y = df['gyroY']

            time = df['timeStamp']
            time_s = time / 1000

            fs = input.fs()


            start_index = np.argmax(
                time >= input.start_time()
            )

            end_index = np.argmin(
                np.abs(
                    time - (
                        np.max(time)
                        - input.end_time()
                    )
                )
            )

            acc_y_filt = acc_y[start_index:end_index]
            acc_z_filt = acc_z[start_index:end_index]
            acc_x_filt = acc_x[start_index:end_index]

            gyro_y_filt = gyro_y[start_index:end_index]
            gyro_z_filt = gyro_z[start_index:end_index]
            gyro_x_filt = gyro_x[start_index:end_index]

            time_filt = time[start_index:end_index]

            # =================================================
            # TOTAL MAGNITUDES
            # =================================================

            Tot_acc_magn = np.sqrt(
                acc_x_filt**2
                + acc_y_filt**2
                + acc_z_filt**2
            )

            Tot_gyro_magn = np.sqrt(
                gyro_x_filt**2
                + gyro_y_filt**2
                + gyro_z_filt**2
            )

            # =================================================
            # STEP DETECTION
            # =================================================

            smoothed_acc_y = savgol_filter(
                acc_y_filt,
                window_length=81,
                polyorder=2
            )

            peaks, _ = find_peaks(
                smoothed_acc_y,
                height=-1,
                distance=150
            )

            selected_peaks = peaks[2:-1]

            num_peaks = len(selected_peaks) - 1

            # =================================================
            # STEP TIME
            # =================================================

            if len(selected_peaks) > 1:

                peak_distances = np.diff(
                    time_filt.iloc[selected_peaks]
                )

                step_time = np.mean(
                    peak_distances
                )

                cv_step_time = (
                    np.std(peak_distances)
                    /
                    np.mean(peak_distances)
                )

                stride_freq = (
                    1000 / (step_time * 2)
                )

            else:

                step_time = np.nan
                cv_step_time = np.nan
                stride_freq = 1.0


            step_total_acc_magnitudes = []
            step_total_gyro_magnitudes = []

            if len(selected_peaks) > 1:

                for i in range(
                    len(selected_peaks) - 1
                ):

                    start_time_step = time_filt.iloc[
                        selected_peaks[i]
                    ]

                    end_time_step = time_filt.iloc[
                        selected_peaks[i + 1]
                    ]

                    segment_mask = (
                        (time_filt >= start_time_step)
                        &
                        (time_filt <= end_time_step)
                    )

                    segment_tot_acc_magn = (
                        Tot_acc_magn[
                            segment_mask
                        ]
                    )

                    segment_tot_gyro_magn = (
                        Tot_gyro_magn[
                            segment_mask
                        ]
                    )

                    if not segment_tot_acc_magn.empty:

                        step_total_acc_magnitudes.append(
                            segment_tot_acc_magn.mean()
                        )

                    if not segment_tot_gyro_magn.empty:

                        step_total_gyro_magnitudes.append(
                            segment_tot_gyro_magn.mean()
                        )

                Acc_magnit_mean = np.mean(
                    step_total_acc_magnitudes
                )

                Gyro_magnit_mean = np.mean(
                    step_total_gyro_magnitudes
                )

            else:

                Acc_magnit_mean = np.nan
                Gyro_magnit_mean = np.nan

            
            cutoff = input.cutoff()

            def butter_lowpass_filter(
                data,
                cutoff,
                fs,
                order=2
            ):

                nyq = 0.5 * fs

                normal_cutoff = cutoff / nyq

                b, a = butter(
                    order,
                    normal_cutoff,
                    btype='low'
                )

                return filtfilt(
                    b,
                    a,
                    data
                )

            acc_x_series = pd.Series(
                butter_lowpass_filter(
                    acc_x_filt,
                    cutoff,
                    fs
                )
            )

            acc_y_series = pd.Series(
                butter_lowpass_filter(
                    acc_y_filt,
                    cutoff,
                    fs
                )
            )

            acc_z_series = pd.Series(
                butter_lowpass_filter(
                    acc_z_filt,
                    cutoff,
                    fs
                )
            )


            gait_dataX = calculate_gait_metrics(
                acc_x_series,
                fs,
                stride_freq,
                'x'
            )

            gait_dataY = calculate_gait_metrics(
                acc_y_series,
                fs,
                stride_freq,
                'y'
            )

            gait_dataZ = calculate_gait_metrics(
                acc_z_series,
                fs,
                stride_freq,
                'z'
            )
            
            spatiotemporal = (
                calculate_spatiotemporal_metrics(
                    acc_y_filt,
                    time_filt,
                    selected_peaks,
                    input.height()
                )
            )
            

            sm_acc_x = savgol_filter(
                acc_x_filt,
                51,
                3
            )

            sm_acc_y = savgol_filter(
                acc_y_filt,
                51,
                3
            )

            sm_acc_z = savgol_filter(
                acc_z_filt,
                51,
                3
            )

            sm_gyro_x = savgol_filter(
                gyro_x_filt,
                51,
                3
            )

            sm_gyro_y = savgol_filter(
                gyro_y_filt,
                51,
                3
            )

            sm_gyro_z = savgol_filter(
                gyro_z_filt,
                51,
                3
            )
            

            time_series_list = [

                (acc_z_filt, 'acc_z_filt'),
                (acc_x_filt, 'acc_x_filt'),
                (acc_y_filt, 'acc_y_filt'),

                (gyro_z_filt, 'gyro_z_filt'),
                (gyro_x_filt, 'gyro_x_filt'),
                (gyro_y_filt, 'gyro_y_filt'),

                (sm_acc_z, 'sm_acc_z'),
                (sm_acc_x, 'sm_acc_x'),
                (sm_acc_y, 'sm_acc_y'),

                (sm_gyro_z, 'sm_gyro_z'),
                (sm_gyro_x, 'sm_gyro_x'),
                (sm_gyro_y, 'sm_gyro_y')
            ]

            lyapunov_results = {}

            for ts, name in time_series_list:

                try:

                    lyap_exponent = calculate_lyapunov(
                        ts,
                        fs,
                        step_time
                    )

                    lyapunov_results[name] = (
                        lyap_exponent
                    )

                except Exception as e:

                    lyapunov_results[name] = (
                        f"Error: {e}"
                    )
          

            results = {

                'raw_data': (
                    time_s,
                    acc_x,
                    acc_y,
                    acc_z,
                    gyro_x,
                    gyro_y,
                    gyro_z,
                    start_index,
                    end_index
                ),

                'filtered_data': (
                    time_filt,
                    acc_x_filt,
                    acc_y_filt,
                    acc_z_filt,
                    gyro_x_filt,
                    gyro_y_filt,
                    gyro_z_filt
                ),

                'smoothed_data': (
                    sm_acc_x,
                    sm_acc_y,
                    sm_acc_z,
                    sm_gyro_x,
                    sm_gyro_y,
                    sm_gyro_z
                ),

                'step_data': (
                    smoothed_acc_y,
                    peaks,
                    selected_peaks
                ),

                'step_metrics': (
                    num_peaks,
                    step_time,
                    cv_step_time,
                    Acc_magnit_mean,
                    Gyro_magnit_mean
                ),

                'spatiotemporal_metrics': (
                    spatiotemporal
                ),

                'gait_metrics': (
                    gait_dataX,
                    gait_dataY,
                    gait_dataZ
                ),

                'lyapunov': (
                    lyapunov_results
                ),

                'Tot_acc_magn': (
                    Tot_acc_magn
                )
            }

            analysis_results.set(results)

            ui.notification_show(
                "Analysis completed successfully!",
                type="success"
            )

        except Exception as e:

            ui.notification_show(
                f"Error during analysis: {str(e)}",
                type="error"
            )
  

    @render.plot
    def raw_plot():

        results = analysis_results.get()

        if results is None:
            return None

        (
            time_s,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            start_index,
            end_index

        ) = results['raw_data']

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(10, 8)
        )

        axes[0].plot(time_s, acc_x, label='Acc X')
        axes[0].plot(time_s, acc_y, label='Acc Y')
        axes[0].plot(time_s, acc_z, label='Acc Z')

        axes[0].set_title(
            'Accelerometer Signals'
        )

        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time_s, gyro_x, label='Gyro X')
        axes[1].plot(time_s, gyro_y, label='Gyro Y')
        axes[1].plot(time_s, gyro_z, label='Gyro Z')

        axes[1].set_title(
            'Gyroscope Signals'
        )

        axes[1].legend()
        axes[1].grid(True)

        time_filt = results['filtered_data'][0]

        Tot_acc_magn = results[
            'Tot_acc_magn'
        ]

        axes[2].plot(
            time_filt,
            Tot_acc_magn,
            label='Total Acceleration'
        )

        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        return fig


    @render.plot
    def step_plot():

        results = analysis_results.get()

        if results is None:
            return None

        (
            smoothed_acc_y,
            peaks,
            selected_peaks

        ) = results['step_data']

        time_filt = results[
            'filtered_data'
        ][0]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            time_filt,
            smoothed_acc_y,
            color='red'
        )

        ax.plot(
            time_filt.iloc[peaks],
            smoothed_acc_y[peaks],
            'x',
            color='blue'
        )

        ax.plot(
            time_filt.iloc[selected_peaks],
            smoothed_acc_y[selected_peaks],
            'o',
            color='green'
        )

        ax.set_title(
            'Step Detection'
        )

        ax.grid(True)

        return fig


    @render.plot
    def filtered_smoothed_plot():

        results = analysis_results.get()

        if results is None:
            return None

        (
            time_filt,
            acc_x_filt,
            acc_y_filt,
            acc_z_filt,
            gyro_x_filt,
            gyro_y_filt,
            gyro_z_filt

        ) = results['filtered_data']

        (
            sm_acc_x,
            sm_acc_y,
            sm_acc_z,
            sm_gyro_x,
            sm_gyro_y,
            sm_gyro_z

        ) = results['smoothed_data']

        fig, axes = plt.subplots(
            6,
            1,
            figsize=(12, 14)
        )

        signals = [

            (acc_x_filt, sm_acc_x, 'Acc X'),
            (acc_y_filt, sm_acc_y, 'Acc Y'),
            (acc_z_filt, sm_acc_z, 'Acc Z'),

            (gyro_x_filt, sm_gyro_x, 'Gyro X'),
            (gyro_y_filt, sm_gyro_y, 'Gyro Y'),
            (gyro_z_filt, sm_gyro_z, 'Gyro Z')
        ]

        for ax, sig in zip(axes, signals):

            raw_sig, smooth_sig, title = sig

            ax.plot(
                time_filt,
                raw_sig,
                alpha=0.6
            )

            ax.plot(
                time_filt,
                smooth_sig,
                color='red'
            )

            ax.set_title(title)
            ax.grid(True)

        plt.tight_layout()

        return fig


    @render.text
    def step_metrics():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        (
            num_peaks,
            step_time,
            cv_step_time,
            Acc_magnit_mean,
            Gyro_magnit_mean

        ) = results['step_metrics']

        text = f"""
STEP METRICS
====================================

Number of steps:
{num_peaks:.0f}

Mean step time:
{step_time:.2f} ms

CV step time:
{cv_step_time:.4f}

Mean step total acceleration magnitude:
{Acc_magnit_mean:.4f} m/s²

Mean step total gyroscope magnitude:
{Gyro_magnit_mean:.4f} rad/s
"""

        return text


    @render.text
    def spatiotemporal_metrics():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        metrics = results[
            'spatiotemporal_metrics'
        ]

        text = f"""
SPATIOTEMPORAL METRICS
====================================

Step Length:
{metrics['step_length']:.4f} m

Gait Speed:
{metrics['gait_speed']:.4f} m/s

Cadence:
{metrics['cadence']:.2f} steps/min

Walk Ratio:
{metrics['walk_ratio']:.6f}

Normalized Walk Ratio:
{metrics['normalized_walk_ratio']:.6f}
"""

        return text


    @render.text
    def gait_metrics_x():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        gait_dataX = results[
            'gait_metrics'
        ][0]

        return format_gait_metrics(
            gait_dataX,
            "X-axis"
        )


    @render.text
    def gait_metrics_y():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        gait_dataY = results[
            'gait_metrics'
        ][1]

        return format_gait_metrics(
            gait_dataY,
            "Y-axis"
        )

    @render.text
    def gait_metrics_z():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        gait_dataZ = results[
            'gait_metrics'
        ][2]

        return format_gait_metrics(
            gait_dataZ,
            "Z-axis"
        )


    @render.text
    def lyapunov_results():

        results = analysis_results.get()

        if results is None:
            return "No analysis results available."

        lyap = results['lyapunov']

        output = (
            "LYAPUNOV EXPONENTS\n"
            + "="*40 + "\n\n"
        )

        for name, value in lyap.items():

            if isinstance(value, str):

                output += f"{name}: {value}\n"

            else:

                output += (
                    f"{name}: "
                    f"{value:.6f}\n"
                )

        return output


def format_gait_metrics(
    gait_data,
    axis_name
):

    output = (
        f"\nGAIT METRICS ({axis_name})\n"
        + "="*40 + "\n\n"
    )

    for col in gait_data.columns:

        value = gait_data[col].iloc[0]

        output += (
            f"{col}: "
            f"{value:.6f}\n"
        )

    return output


def calculate_spatiotemporal_metrics(
    acc_vertical,
    time_filt,
    selected_peaks,
    subject_height
):

    metrics = {}

    if len(selected_peaks) < 2:

        metrics["step_length"] = np.nan
        metrics["gait_speed"] = np.nan
        metrics["cadence"] = np.nan
        metrics["walk_ratio"] = np.nan
        metrics["normalized_walk_ratio"] = np.nan

        return metrics

    peak_times = (
        time_filt.iloc[
            selected_peaks
        ].values
    )

    step_intervals = (
        np.diff(peak_times)
        / 1000.0
    )

    mean_step_time = np.mean(
        step_intervals
    )

    cadence = (
        60 / mean_step_time
    )

    vertical_rms = np.sqrt(
        np.mean(acc_vertical**2)
    )

    step_length = (
        0.25
        * subject_height
        * np.sqrt(vertical_rms)
        * np.sqrt(mean_step_time)
    )

    step_length = np.clip(
        step_length,
        0.2,
        1.2
    )

    gait_speed = (
        step_length
        / mean_step_time
    )

    walk_ratio = (
        step_length
        / cadence
    )

    normalized_walk_ratio = (
        walk_ratio
        / subject_height
    )

    metrics["step_length"] = step_length
    metrics["gait_speed"] = gait_speed
    metrics["cadence"] = cadence
    metrics["walk_ratio"] = walk_ratio
    metrics["normalized_walk_ratio"] = normalized_walk_ratio

    return metrics


def harmonic_ratio(data):

    fft_result = np.fft.fft(data)

    f1 = np.abs(fft_result[1])
    f2 = np.abs(fft_result[2])

    if f2 == 0:
        return np.nan

    return f1 / f2

def harmonic_ratio_power(signal, fs):

    frequencies, power_spectrum = welch(
        signal,
        fs=fs
    )

    fundamental_freq_idx = np.argmax(
        power_spectrum
    )

    second_harmonic_idx = np.argmin(
        np.abs(
            frequencies
            - 2 * frequencies[
                fundamental_freq_idx
            ]
        )
    )

    return (
        power_spectrum[
            second_harmonic_idx
        ]
        /
        power_spectrum[
            fundamental_freq_idx
        ]
    )

def HR_even_odd(
    data,
    sampling_rate,
    stride_freq
):

    N = len(data)

    freqs = fftfreq(
        N,
        d=1/sampling_rate
    )[:N//2]

    spectrum = np.abs(
        fft(data)
    )[:N//2]

    harmonics = np.array([
        i * stride_freq
        for i in range(1, 21)
    ])

    idx = [
        np.argmin(np.abs(freqs - h))
        for h in harmonics
    ]

    amplitudes = spectrum[idx]

    even_harmonics = amplitudes[1::2]
    odd_harmonics = amplitudes[0::2]

    return (
        np.sum(even_harmonics)
        /
        np.sum(odd_harmonics)
    )

def root_mean_square(data):

    return np.sqrt(np.mean(data**2))

def coefficient_variation(data):

    return np.std(data) / np.mean(data)

def sparc(
    data,
    sampling_rate,
    omega_c=10
):

    N = len(data)

    acc_fft = np.abs(
        fft(data)
    )[:N//2]

    freqs = fftfreq(
        N,
        d=1/sampling_rate
    )[:N//2]

    acc_fft /= np.max(acc_fft)

    mask = freqs <= omega_c

    S = acc_fft[mask]
    omega = freqs[mask]

    dS = np.gradient(S, omega)

    integrand = np.sqrt(
        (1/omega_c)**2 + dS**2
    )

    delta = np.mean(np.diff(omega))

    return -np.sum(integrand) * delta

def ldlj(data, sampling_rate):

    dt = 1 / sampling_rate

    jerk = np.gradient(data, dt)

    jerk_squared_mean = np.mean(
        jerk**2
    )

    duration = len(data) * dt

    a_peak = np.max(np.abs(data))

    return -np.log(
        (duration / (a_peak**2))
        * jerk_squared_mean
    )

def calculate_gait_metrics(
    acc_series,
    fs,
    stride_freq,
    axis
):

    gait_data = pd.DataFrame(index=[0])

    gait_data[f'HR_{axis}'] = harmonic_ratio(
        acc_series.values
    )

    gait_data[f'HRp_{axis}'] = (
        harmonic_ratio_power(
            acc_series.values,
            fs
        )
    )

    hreo_val = HR_even_odd(
        acc_series.values,
        fs,
        stride_freq
    )
    
    if axis.lower() == 'x':
        gait_data[f'HReo_{axis}'] = (1 / hreo_val) if hreo_val != 0 else np.nan
    else:
        gait_data[f'HReo_{axis}'] = hreo_val

    gait_data[f'RMS_{axis}'] = (
        root_mean_square(
            acc_series.values
        )
    )

    gait_data[f'CV_{axis}'] = (
        coefficient_variation(
            acc_series.values
        )
    )

    gait_data[f'SPARC_{axis}'] = (
        sparc(
            acc_series.values,
            fs
        )
    )

    gait_data[f'LDLJ_{axis}'] = (
        ldlj(
            acc_series.values,
            fs
        )
    )

    return gait_data

def calculate_lyapunov(
    time_series,
    fs,
    step_time
):

    if np.isnan(step_time):

        max_delay = 100

    else:

        max_delay_ms = (
            step_time * 2.5
        )

        max_delay = int(
            (max_delay_ms / 1000)
            * fs
        )

    max_delay = min(
        max_delay,
        len(time_series) - 1
    )

    lags = np.arange(
        1,
        max_delay + 1
    )

    ts_array = np.asarray(
        time_series
    )

    mutual_information_values = []

    for lag in lags:

        mi = mutual_info_score(
            ts_array[:-lag],
            ts_array[lag:]
        )

        mutual_information_values.append(mi)

    if len(mutual_information_values) < 3:

        optimal_lag = 10

    else:

        smoothed_mi = savgol_filter(
            mutual_information_values,
            window_length=11,
            polyorder=3
        )

        min_indices, _ = find_peaks(
            -smoothed_mi
        )

        if len(min_indices) > 0:

            optimal_lag = (
                lags[min_indices[0]]
            )

        else:

            optimal_lag = 10

    lyap_exponent = (
        nolds.lyap_r(
            ts_array,
            emb_dim=7,
            tau=optimal_lag,
            min_tsep=250
        )
        * fs
    )

    return lyap_exponent

app = App(app_ui, server)
