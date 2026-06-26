from shiny import App, render, ui, reactive

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import base64
import tempfile
import os

from scipy.signal import (
    welch,
    butter,
    filtfilt,
    savgol_filter,
    find_peaks
)

from scipy.fft import fft, fftfreq
from sklearn.metrics import mutual_info_score

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import nolds
import warnings

warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='sklearn.metrics'
)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

app_ui = ui.page_sidebar(

    ui.head_content(
        ui.tags.style("""
        .progress-container {
            margin: 10px 0;
            display: none;
        }
        .progress-container.visible {
            display: block;
        }
        .progress-bar-wrap {
            background: #e9ecef;
            border-radius: 6px;
            height: 20px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #0d6efd, #0a58ca);
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        .progress-label {
            font-size: 0.8em;
            color: #555;
            margin-top: 4px;
            min-height: 1.2em;
        }
        .sidebar-section {
            margin-bottom: 8px;
        }
        """)
    ),

    ui.sidebar(

        ui.div(
            ui.input_file(
                "file",
                "Upload CSV File",
                accept=[".csv"],
                multiple=False
            ),
            class_="sidebar-section"
        ),

        ui.div(
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
            id="analysis_params"
        ),

        ui.div(
            ui.input_action_button(
                "analyze",
                "▶  Run Analysis",
                class_="btn-primary w-100",
                disabled=True
            ),
            class_="sidebar-section"
        ),

        ui.div(
            ui.output_ui("progress_ui"),
            class_="sidebar-section"
        ),

        ui.div(
            ui.input_action_button(
                "create_report",
                "📄  Create Report",
                class_="btn-success w-100",
                disabled=True
            ),
            class_="sidebar-section"
        ),

        ui.output_ui("report_download_ui"),

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

# ─────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────

def server(input, output, session):

    raw_data_loaded  = reactive.value(None)   # only raw signals
    analysis_results = reactive.value(None)   # full analysis
    progress_state   = reactive.value({"pct": 0, "label": "", "visible": False})
    report_path      = reactive.value(None)

    # ── 1. Load file and show raw plots immediately ──────────────────

    @reactive.effect
    @reactive.event(input.file)
    def _load_file():

        file_info = input.file()

        if file_info is None:
            return

        try:
            df = pd.read_csv(file_info[0]["datapath"])

            acc_z = -df['accZ']
            acc_x =  df['accX']
            acc_y =  df['accY'] - 9.81

            gyro_z = df['gyroZ']
            gyro_x = df['gyroX']
            gyro_y = df['gyroY']

            time   = df['timeStamp']
            time_s = time / 1000

            raw_data_loaded.set({
                'time':   time,
                'time_s': time_s,
                'acc_x':  acc_x,
                'acc_y':  acc_y,
                'acc_z':  acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
            })

            # enable Run Analysis button
            ui.update_action_button("analyze", disabled=False)

            ui.notification_show(
                "File loaded – set parameters and click Run Analysis.",
                type="message"
            )

        except Exception as e:
            ui.notification_show(
                f"Error loading file: {str(e)}",
                type="error"
            )

    # ── 2. Full analysis on button click ─────────────────────────────

    @reactive.effect
    @reactive.event(input.analyze)
    def _run_analysis():

        raw = raw_data_loaded.get()

        if raw is None:
            ui.notification_show(
                "Please upload a CSV file first.",
                type="warning"
            )
            return

        def set_progress(pct, label):
            progress_state.set({
                "pct": pct,
                "label": label,
                "visible": True
            })

        try:

            set_progress(5, "Reading signals…")

            time   = raw['time']
            time_s = raw['time_s']
            acc_x  = raw['acc_x']
            acc_y  = raw['acc_y']
            acc_z  = raw['acc_z']
            gyro_x = raw['gyro_x']
            gyro_y = raw['gyro_y']
            gyro_z = raw['gyro_z']

            fs = input.fs()

            # ── trim ────────────────────────────────
            set_progress(10, "Trimming data…")

            start_index = np.argmax(time >= input.start_time())
            end_index   = np.argmin(np.abs(time - (np.max(time) - input.end_time())))

            acc_y_filt  = acc_y[start_index:end_index]
            acc_z_filt  = acc_z[start_index:end_index]
            acc_x_filt  = acc_x[start_index:end_index]

            gyro_y_filt = gyro_y[start_index:end_index]
            gyro_z_filt = gyro_z[start_index:end_index]
            gyro_x_filt = gyro_x[start_index:end_index]

            time_filt = time[start_index:end_index]

            # ── magnitudes ──────────────────────────
            set_progress(18, "Computing magnitudes…")

            Tot_acc_magn = np.sqrt(
                acc_x_filt**2 + acc_y_filt**2 + acc_z_filt**2
            )

            Tot_gyro_magn = np.sqrt(
                gyro_x_filt**2 + gyro_y_filt**2 + gyro_z_filt**2
            )

            # ── step detection ──────────────────────
            set_progress(28, "Detecting steps…")

            smoothed_acc_y = savgol_filter(
                acc_y_filt, window_length=81, polyorder=2
            )

            peaks, _ = find_peaks(
                smoothed_acc_y, height=-1, distance=150
            )

            selected_peaks = peaks[2:-1]
            num_peaks      = len(selected_peaks) - 1

            # ── step timing ─────────────────────────
            set_progress(35, "Computing step timing…")

            if len(selected_peaks) > 1:
                peak_distances = np.diff(time_filt.iloc[selected_peaks])
                step_time      = np.mean(peak_distances)
                cv_step_time   = np.std(peak_distances) / np.mean(peak_distances)
                stride_freq    = 1000 / (step_time * 2)
            else:
                step_time    = np.nan
                cv_step_time = np.nan
                stride_freq  = 1.0

            # ── per-step magnitudes ─────────────────
            set_progress(42, "Averaging per-step magnitudes…")

            step_total_acc_magnitudes  = []
            step_total_gyro_magnitudes = []

            if len(selected_peaks) > 1:
                for i in range(len(selected_peaks) - 1):
                    s_t = time_filt.iloc[selected_peaks[i]]
                    e_t = time_filt.iloc[selected_peaks[i + 1]]
                    mask = (time_filt >= s_t) & (time_filt <= e_t)
                    seg_acc  = Tot_acc_magn[mask]
                    seg_gyro = Tot_gyro_magn[mask]
                    if not seg_acc.empty:
                        step_total_acc_magnitudes.append(seg_acc.mean())
                    if not seg_gyro.empty:
                        step_total_gyro_magnitudes.append(seg_gyro.mean())

                Acc_magnit_mean  = np.mean(step_total_acc_magnitudes)
                Gyro_magnit_mean = np.mean(step_total_gyro_magnitudes)
            else:
                Acc_magnit_mean  = np.nan
                Gyro_magnit_mean = np.nan

            # ── low-pass filter ─────────────────────
            set_progress(50, "Low-pass filtering…")

            cutoff = input.cutoff()

            def butter_lowpass_filter(data, cutoff, fs, order=2):
                nyq = 0.5 * fs
                b, a = butter(order, cutoff / nyq, btype='low')
                return filtfilt(b, a, data)

            acc_x_series = pd.Series(butter_lowpass_filter(acc_x_filt, cutoff, fs))
            acc_y_series = pd.Series(butter_lowpass_filter(acc_y_filt, cutoff, fs))
            acc_z_series = pd.Series(butter_lowpass_filter(acc_z_filt, cutoff, fs))

            # ── gait metrics ────────────────────────
            set_progress(58, "Computing gait metrics (X axis)…")
            gait_dataX = calculate_gait_metrics(acc_x_series, fs, stride_freq, 'x')

            set_progress(65, "Computing gait metrics (Y axis)…")
            gait_dataY = calculate_gait_metrics(acc_y_series, fs, stride_freq, 'y')

            set_progress(70, "Computing gait metrics (Z axis)…")
            gait_dataZ = calculate_gait_metrics(acc_z_series, fs, stride_freq, 'z')

            # ── spatiotemporal ──────────────────────
            set_progress(75, "Computing spatiotemporal metrics…")

            spatiotemporal = calculate_spatiotemporal_metrics(
                acc_y_filt, time_filt, selected_peaks, input.height()
            )

            # ── smoothing (for plots) ───────────────
            set_progress(80, "Smoothing signals…")

            sm_acc_x  = savgol_filter(acc_x_filt, 51, 3)
            sm_acc_y  = savgol_filter(acc_y_filt, 51, 3)
            sm_acc_z  = savgol_filter(acc_z_filt, 51, 3)

            sm_gyro_x = savgol_filter(gyro_x_filt, 51, 3)
            sm_gyro_y = savgol_filter(gyro_y_filt, 51, 3)
            sm_gyro_z = savgol_filter(gyro_z_filt, 51, 3)

            # ── Lyapunov ────────────────────────────
            set_progress(86, "Computing Lyapunov exponents…")

            time_series_list = [
                (acc_z_filt,  'acc_z_filt'),
                (acc_x_filt,  'acc_x_filt'),
                (acc_y_filt,  'acc_y_filt'),
                (gyro_z_filt, 'gyro_z_filt'),
                (gyro_x_filt, 'gyro_x_filt'),
                (gyro_y_filt, 'gyro_y_filt'),
                (sm_acc_z,    'sm_acc_z'),
                (sm_acc_x,    'sm_acc_x'),
                (sm_acc_y,    'sm_acc_y'),
                (sm_gyro_z,   'sm_gyro_z'),
                (sm_gyro_x,   'sm_gyro_x'),
                (sm_gyro_y,   'sm_gyro_y'),
            ]

            lyapunov_results_dict = {}

            for ts, name in time_series_list:
                try:
                    lyap_exponent = calculate_lyapunov(ts, fs, step_time)
                    lyapunov_results_dict[name] = lyap_exponent
                except Exception as e:
                    lyapunov_results_dict[name] = f"Error: {e}"

            # ── store results ───────────────────────
            set_progress(97, "Finalising results…")

            results = {

                'raw_data': (
                    time_s, acc_x, acc_y, acc_z,
                    gyro_x, gyro_y, gyro_z,
                    start_index, end_index
                ),

                'filtered_data': (
                    time_filt,
                    acc_x_filt, acc_y_filt, acc_z_filt,
                    gyro_x_filt, gyro_y_filt, gyro_z_filt
                ),

                'smoothed_data': (
                    sm_acc_x, sm_acc_y, sm_acc_z,
                    sm_gyro_x, sm_gyro_y, sm_gyro_z
                ),

                'step_data': (
                    smoothed_acc_y, peaks, selected_peaks
                ),

                'step_metrics': (
                    num_peaks, step_time, cv_step_time,
                    Acc_magnit_mean, Gyro_magnit_mean
                ),

                'spatiotemporal_metrics': spatiotemporal,

                'gait_metrics': (gait_dataX, gait_dataY, gait_dataZ),

                'lyapunov': lyapunov_results_dict,

                'Tot_acc_magn': Tot_acc_magn
            }

            analysis_results.set(results)

            # enable report button
            ui.update_action_button("create_report", disabled=False)

            set_progress(100, "Analysis complete!")

            ui.notification_show(
                "Analysis completed successfully!",
                type="success"
            )

        except Exception as e:
            progress_state.set({"pct": 0, "label": f"Error: {e}", "visible": True})
            ui.notification_show(
                f"Error during analysis: {str(e)}",
                type="error"
            )

    # ── 3. Create PDF report ──────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.create_report)
    def _create_report():

        results = analysis_results.get()

        if results is None:
            return

        progress_state.set({"pct": 10, "label": "Generating report…", "visible": True})

        try:
            pdf_path = generate_pdf_report(results)
            report_path.set(pdf_path)
            progress_state.set({"pct": 100, "label": "Report ready!", "visible": True})

            ui.notification_show(
                "Report created! Click 'Download Report' to save.",
                type="success"
            )

        except Exception as e:
            progress_state.set({"pct": 0, "label": f"Report error: {e}", "visible": True})
            ui.notification_show(
                f"Error creating report: {str(e)}",
                type="error"
            )

    # ── Progress UI ───────────────────────────────────────────────────

    @render.ui
    def progress_ui():
        state = progress_state.get()
        if not state["visible"]:
            return ui.div()

        pct   = state["pct"]
        label = state["label"]

        return ui.div(
            ui.div(
                ui.div(
                    style=f"width:{pct}%; height:100%; background:linear-gradient(90deg,#0d6efd,#0a58ca); border-radius:6px; transition:width 0.3s ease;"
                ),
                style="background:#e9ecef; border-radius:6px; height:20px; overflow:hidden;"
            ),
            ui.div(
                f"{label}  ({pct}%)",
                style="font-size:0.8em; color:#555; margin-top:4px;"
            ),
            style="margin:8px 0;"
        )

    # ── Report download link ──────────────────────────────────────────

    @render.ui
    def report_download_ui():
        path = report_path.get()
        if path is None or not os.path.exists(path):
            return ui.div()

        # encode as base64 data-URI for download
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        return ui.div(
            ui.tags.a(
                "⬇  Download Report",
                href=f"data:application/pdf;base64,{b64}",
                download="gait_analysis_report.pdf",
                class_="btn btn-outline-success btn-sm w-100",
                style="margin-top:4px;"
            )
        )

    # ── Plots ─────────────────────────────────────────────────────────

    @render.plot
    def raw_plot():
        # Show raw data as soon as file is loaded
        raw = raw_data_loaded.get()
        if raw is None:
            return None

        time_s = raw['time_s']
        acc_x  = raw['acc_x']
        acc_y  = raw['acc_y']
        acc_z  = raw['acc_z']
        gyro_x = raw['gyro_x']
        gyro_y = raw['gyro_y']
        gyro_z = raw['gyro_z']

        # After analysis we also have trim markers and magnitude
        results = analysis_results.get()

        nrows = 3 if results else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows))

        axes[0].plot(time_s, acc_x, label='Acc X')
        axes[0].plot(time_s, acc_y, label='Acc Y')
        axes[0].plot(time_s, acc_z, label='Acc Z')

        if results:
            (_, _, _, _, _, _, _, start_index, end_index) = results['raw_data']
            axes[0].axvline(x=time_s.iloc[start_index], color='k',       linestyle='--', label='Start')
            axes[0].axvline(x=time_s.iloc[end_index],   color='dimgray', linestyle='--', label='End')

        axes[0].set_title('Accelerometer Signals')
        axes[0].set_ylabel('m/s²')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time_s, gyro_x, label='Gyro X')
        axes[1].plot(time_s, gyro_y, label='Gyro Y')
        axes[1].plot(time_s, gyro_z, label='Gyro Z')

        if results:
            axes[1].axvline(x=time_s.iloc[start_index], color='k',       linestyle='--', label='Start')
            axes[1].axvline(x=time_s.iloc[end_index],   color='dimgray', linestyle='--', label='End')

        axes[1].set_title('Gyroscope Signals')
        axes[1].set_ylabel('rad/s')
        axes[1].legend()
        axes[1].grid(True)

        if results:
            time_filt    = results['filtered_data'][0]
            Tot_acc_magn = results['Tot_acc_magn']
            axes[2].plot(time_filt, Tot_acc_magn, label='Total Acceleration Magnitude', color='purple')
            axes[2].set_title('Total Acceleration Magnitude')
            axes[2].set_ylabel('m/s²')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        return fig

    @render.plot
    def step_plot():
        results = analysis_results.get()
        if results is None:
            return None

        (smoothed_acc_y, peaks, selected_peaks) = results['step_data']
        time_filt = results['filtered_data'][0]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_filt, smoothed_acc_y, color='red')
        ax.plot(time_filt.iloc[peaks],          smoothed_acc_y[peaks],          'x', color='blue',  label='All peaks')
        ax.plot(time_filt.iloc[selected_peaks], smoothed_acc_y[selected_peaks], 'o', color='green', label='Selected peaks')
        ax.set_title('Step Detection')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Acc Y (smoothed) m/s²')
        ax.legend()
        ax.grid(True)
        return fig

    @render.plot
    def filtered_smoothed_plot():
        results = analysis_results.get()
        if results is None:
            return None

        (time_filt, acc_x_filt, acc_y_filt, acc_z_filt,
         gyro_x_filt, gyro_y_filt, gyro_z_filt) = results['filtered_data']

        (sm_acc_x, sm_acc_y, sm_acc_z,
         sm_gyro_x, sm_gyro_y, sm_gyro_z) = results['smoothed_data']

        fig, axes = plt.subplots(6, 1, figsize=(12, 14))

        signals = [
            (acc_x_filt,  sm_acc_x,  'Acc X'),
            (acc_y_filt,  sm_acc_y,  'Acc Y'),
            (acc_z_filt,  sm_acc_z,  'Acc Z'),
            (gyro_x_filt, sm_gyro_x, 'Gyro X'),
            (gyro_y_filt, sm_gyro_y, 'Gyro Y'),
            (gyro_z_filt, sm_gyro_z, 'Gyro Z'),
        ]

        for ax, (raw_sig, smooth_sig, title) in zip(axes, signals):
            ax.plot(time_filt, raw_sig, alpha=0.6)
            ax.plot(time_filt, smooth_sig, color='red')
            ax.set_title(title)
            ax.grid(True)

        plt.tight_layout()
        return fig

    # ── Text outputs ──────────────────────────────────────────────────

    @render.text
    def step_metrics():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."

        (num_peaks, step_time, cv_step_time,
         Acc_magnit_mean, Gyro_magnit_mean) = results['step_metrics']

        return f"""
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

    @render.text
    def spatiotemporal_metrics():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."

        m = results['spatiotemporal_metrics']

        return f"""
SPATIOTEMPORAL METRICS
====================================

Step Length:
{m['step_length']:.4f} m

Gait Speed:
{m['gait_speed']:.4f} m/s

Cadence:
{m['cadence']:.2f} steps/min

Walk Ratio:
{m['walk_ratio']:.6f}

Normalized Walk Ratio:
{m['normalized_walk_ratio']:.6f}
"""

    @render.text
    def gait_metrics_x():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."
        return format_gait_metrics(results['gait_metrics'][0], "X-axis")

    @render.text
    def gait_metrics_y():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."
        return format_gait_metrics(results['gait_metrics'][1], "Y-axis")

    @render.text
    def gait_metrics_z():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."
        return format_gait_metrics(results['gait_metrics'][2], "Z-axis")

    @render.text
    def lyapunov_results():
        results = analysis_results.get()
        if results is None:
            return "No analysis results available."

        lyap   = results['lyapunov']
        output = "LYAPUNOV EXPONENTS\n" + "="*40 + "\n\n"

        for name, value in lyap.items():
            if isinstance(value, str):
                output += f"{name}: {value}\n"
            else:
                output += f"{name}: {value:.6f}\n"

        return output


# ─────────────────────────────────────────────
# HELPER FUNCTIONS (unchanged logic)
# ─────────────────────────────────────────────

def format_gait_metrics(gait_data, axis_name):
    output = f"\nGAIT METRICS ({axis_name})\n" + "="*40 + "\n\n"
    for col in gait_data.columns:
        output += f"{col}: {gait_data[col].iloc[0]:.6f}\n"
    return output


def calculate_spatiotemporal_metrics(acc_vertical, time_filt, selected_peaks, subject_height):
    metrics = {}

    if len(selected_peaks) < 2:
        for k in ("step_length", "gait_speed", "cadence", "walk_ratio", "normalized_walk_ratio"):
            metrics[k] = np.nan
        return metrics

    peak_times     = time_filt.iloc[selected_peaks].values
    step_intervals = np.diff(peak_times) / 1000.0
    mean_step_time = np.mean(step_intervals)
    cadence        = 60 / mean_step_time
    vertical_rms   = np.sqrt(np.mean(acc_vertical**2))
    step_length    = np.clip(
        0.25 * subject_height * np.sqrt(vertical_rms) * np.sqrt(mean_step_time),
        0.2, 1.2
    )
    gait_speed             = step_length / mean_step_time
    walk_ratio             = step_length / cadence
    normalized_walk_ratio  = walk_ratio / subject_height

    metrics["step_length"]            = step_length
    metrics["gait_speed"]             = gait_speed
    metrics["cadence"]                = cadence
    metrics["walk_ratio"]             = walk_ratio
    metrics["normalized_walk_ratio"]  = normalized_walk_ratio

    return metrics


def harmonic_ratio(data):
    fft_result = np.fft.fft(data)
    f1 = np.abs(fft_result[1])
    f2 = np.abs(fft_result[2])
    return np.nan if f2 == 0 else f1 / f2


def harmonic_ratio_power(signal, fs):
    frequencies, power_spectrum = welch(signal, fs=fs)
    fundamental_freq_idx  = np.argmax(power_spectrum)
    second_harmonic_idx   = np.argmin(np.abs(frequencies - 2 * frequencies[fundamental_freq_idx]))
    return power_spectrum[second_harmonic_idx] / power_spectrum[fundamental_freq_idx]


def HR_even_odd(data, sampling_rate, stride_freq):
    N        = len(data)
    freqs    = fftfreq(N, d=1/sampling_rate)[:N//2]
    spectrum = np.abs(fft(data))[:N//2]
    harmonics = np.array([i * stride_freq for i in range(1, 21)])
    idx       = [np.argmin(np.abs(freqs - h)) for h in harmonics]
    amplitudes = spectrum[idx]
    even_harmonics = amplitudes[1::2]
    odd_harmonics  = amplitudes[0::2]
    return np.sum(even_harmonics) / np.sum(odd_harmonics)


def root_mean_square(data):
    return np.sqrt(np.mean(data**2))


def coefficient_variation(data):
    return np.std(data) / np.mean(data)


def sparc(data, sampling_rate, omega_c=10):
    N       = len(data)
    acc_fft = np.abs(fft(data))[:N//2]
    freqs   = fftfreq(N, d=1/sampling_rate)[:N//2]
    acc_fft /= np.max(acc_fft)
    mask    = freqs <= omega_c
    S, omega = acc_fft[mask], freqs[mask]
    dS      = np.gradient(S, omega)
    integrand = np.sqrt((1/omega_c)**2 + dS**2)
    return -np.sum(integrand) * np.mean(np.diff(omega))


def ldlj(data, sampling_rate):
    dt               = 1 / sampling_rate
    jerk             = np.gradient(data, dt)
    jerk_squared_mean = np.mean(jerk**2)
    duration         = len(data) * dt
    a_peak           = np.max(np.abs(data))
    return -np.log((duration / (a_peak**2)) * jerk_squared_mean)


def calculate_gait_metrics(acc_series, fs, stride_freq, axis):
    gait_data = pd.DataFrame(index=[0])

    gait_data[f'HR_{axis}']   = harmonic_ratio(acc_series.values)
    gait_data[f'HRp_{axis}']  = harmonic_ratio_power(acc_series.values, fs)

    hreo_val = HR_even_odd(acc_series.values, fs, stride_freq)

    # For the mediolateral axis (x), invert the ratio
    if axis.lower() == 'x':
        gait_data[f'HReo_{axis}'] = (1 / hreo_val) if hreo_val != 0 else np.nan
    else:
        gait_data[f'HReo_{axis}'] = hreo_val

    gait_data[f'RMS_{axis}']  = root_mean_square(acc_series.values)
    gait_data[f'CV_{axis}']   = coefficient_variation(acc_series.values)
    gait_data[f'SPARC_{axis}'] = sparc(acc_series.values, fs)
    gait_data[f'LDLJ_{axis}'] = ldlj(acc_series.values, fs)

    return gait_data


def calculate_lyapunov(time_series, fs, step_time):
    if np.isnan(step_time):
        max_delay = 100
    else:
        max_delay = int((step_time * 2.5 / 1000) * fs)

    max_delay = min(max_delay, len(time_series) - 1)
    lags      = np.arange(1, max_delay + 1)
    ts_array  = np.asarray(time_series)

    mutual_information_values = []
    for lag in lags:
        mi = mutual_info_score(ts_array[:-lag], ts_array[lag:])
        mutual_information_values.append(mi)

    if len(mutual_information_values) < 3:
        optimal_lag = 10
    else:
        smoothed_mi  = savgol_filter(mutual_information_values, window_length=11, polyorder=3)
        min_indices, _ = find_peaks(-smoothed_mi)
        optimal_lag  = lags[min_indices[0]] if len(min_indices) > 0 else 10

    lyap_exponent = nolds.lyap_r(
        ts_array, emb_dim=7, tau=optimal_lag, min_tsep=250
    ) * fs

    return lyap_exponent


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────

def fig_to_image_obj(fig, dpi=120):
    """Convert a matplotlib figure to a ReportLab Image object."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_pdf_report(results):
    """Build the PDF report and return its file path."""

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix='.pdf', prefix='gait_report_'
    )
    pdf_path = tmp.name
    tmp.close()

    PAGE_W, PAGE_H = A4
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=6,
        textColor=colors.HexColor('#1a1a2e'),
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=18
    )
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading1'],
        fontSize=13,
        textColor=colors.HexColor('#0d6efd'),
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    metric_label_style = ParagraphStyle(
        'MetricLabel',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#444'),
    )
    metric_val_style = ParagraphStyle(
        'MetricVal',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.black,
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=8
    )

    story = []
    avail_w = PAGE_W - 4*cm   # usable width

    # ── Title page section ──────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Gait Analysis Report", title_style))
    story.append(Paragraph("IMU-based Walking Assessment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0d6efd'), spaceAfter=16))

    # ── Helper: build a 2-col metrics table ─────────────────────────
    def metrics_table(rows):
        """rows: list of (label, value_str)"""
        table_data = []
        for label, value in rows:
            table_data.append([
                Paragraph(label, metric_label_style),
                Paragraph(str(value), metric_val_style)
            ])
        t = Table(table_data, colWidths=[avail_w * 0.65, avail_w * 0.35])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1),
             [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID',       (0, 0), (-1, -1), 0.4, colors.HexColor('#dee2e6')),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('LEFTPADDING',  (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING',   (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        return t

    # ── SECTION 1: Accelerometer plot ──────────────────────────────
    story.append(Paragraph("1. Accelerometer Signals", section_style))

    (time_s, acc_x, acc_y, acc_z,
     gyro_x, gyro_y, gyro_z,
     start_index, end_index) = results['raw_data']

    fig_acc, ax_acc = plt.subplots(figsize=(8, 3.5))
    ax_acc.plot(time_s, acc_x, label='Acc X')
    ax_acc.plot(time_s, acc_y, label='Acc Y')
    ax_acc.plot(time_s, acc_z, label='Acc Z')
    ax_acc.axvline(x=time_s.iloc[start_index], color='k',       linestyle='--', label='Start', linewidth=0.9)
    ax_acc.axvline(x=time_s.iloc[end_index],   color='dimgray', linestyle='--', label='End',   linewidth=0.9)
    ax_acc.set_xlabel('Time (s)')
    ax_acc.set_ylabel('m/s²')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.4)
    plt.tight_layout()

    buf_acc = fig_to_image_obj(fig_acc)
    story.append(Image(buf_acc, width=avail_w, height=avail_w * 3.5/8))
    story.append(Paragraph("Figure 1 – Raw accelerometer signals with analysis window delimiters.", caption_style))

    # ── SECTION 2: Gyroscope plot ───────────────────────────────────
    story.append(Paragraph("2. Gyroscope Signals", section_style))

    fig_gyro, ax_gyro = plt.subplots(figsize=(8, 3.5))
    ax_gyro.plot(time_s, gyro_x, label='Gyro X')
    ax_gyro.plot(time_s, gyro_y, label='Gyro Y')
    ax_gyro.plot(time_s, gyro_z, label='Gyro Z')
    ax_gyro.axvline(x=time_s.iloc[start_index], color='k',       linestyle='--', label='Start', linewidth=0.9)
    ax_gyro.axvline(x=time_s.iloc[end_index],   color='dimgray', linestyle='--', label='End',   linewidth=0.9)
    ax_gyro.set_xlabel('Time (s)')
    ax_gyro.set_ylabel('rad/s')
    ax_gyro.legend(fontsize=8)
    ax_gyro.grid(True, alpha=0.4)
    plt.tight_layout()

    buf_gyro = fig_to_image_obj(fig_gyro)
    story.append(Image(buf_gyro, width=avail_w, height=avail_w * 3.5/8))
    story.append(Paragraph("Figure 2 – Raw gyroscope signals with analysis window delimiters.", caption_style))

    # ── SECTION 3: Total acceleration magnitude ─────────────────────
    story.append(Paragraph("3. Total Acceleration Magnitude", section_style))

    time_filt    = results['filtered_data'][0]
    Tot_acc_magn = results['Tot_acc_magn']

    fig_mag, ax_mag = plt.subplots(figsize=(8, 2.8))
    ax_mag.plot(time_filt, Tot_acc_magn, color='purple', linewidth=0.9)
    ax_mag.set_xlabel('Time (ms)')
    ax_mag.set_ylabel('m/s²')
    ax_mag.grid(True, alpha=0.4)
    plt.tight_layout()

    buf_mag = fig_to_image_obj(fig_mag)
    story.append(Image(buf_mag, width=avail_w, height=avail_w * 2.8/8))
    story.append(Paragraph("Figure 3 – Total acceleration magnitude over the analysis window.", caption_style))

    story.append(PageBreak())

    # ── SECTION 4: Step Metrics ─────────────────────────────────────
    story.append(Paragraph("4. Step Metrics", section_style))

    (num_peaks, step_time, cv_step_time,
     Acc_magnit_mean, Gyro_magnit_mean) = results['step_metrics']

    story.append(metrics_table([
        ("Number of Steps",                        f"{num_peaks:.0f}"),
        ("Mean Step Time (ms)",                    f"{step_time:.2f}"),
        ("CV Step Time",                           f"{cv_step_time:.4f}"),
        ("Mean Step Total Acc. Magnitude (m/s²)",  f"{Acc_magnit_mean:.4f}"),
        ("Mean Step Total Gyro Magnitude (rad/s)", f"{Gyro_magnit_mean:.4f}"),
    ]))

    # ── SECTION 5: Spatiotemporal Metrics ───────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("5. Spatiotemporal Metrics", section_style))

    m = results['spatiotemporal_metrics']

    story.append(metrics_table([
        ("Step Length (m)",       f"{m['step_length']:.4f}"),
        ("Gait Speed (m/s)",      f"{m['gait_speed']:.4f}"),
        ("Cadence (steps/min)",   f"{m['cadence']:.2f}"),
        ("Walk Ratio",            f"{m['walk_ratio']:.6f}"),
        ("Normalized Walk Ratio", f"{m['normalized_walk_ratio']:.6f}"),
    ]))

    story.append(PageBreak())

    # ── SECTION 6–8: Gait Metrics per axis ─────────────────────────
    axis_labels = ["X-axis (Mediolateral)", "Y-axis (Vertical)", "Z-axis (Anteroposterior)"]
    gait_all    = results['gait_metrics']

    for idx, (gait_data, axis_label) in enumerate(zip(gait_all, axis_labels), start=6):
        story.append(Paragraph(f"{idx}. Gait Metrics – {axis_label}", section_style))

        rows = []
        for col in gait_data.columns:
            val = gait_data[col].iloc[0]
            rows.append((col, f"{val:.6f}"))

        story.append(metrics_table(rows))
        story.append(Spacer(1, 0.3*cm))

    # ── Footer note ─────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey, spaceBefore=12))
    story.append(Paragraph(
        "Report generated automatically by the Gait Analysis Application.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7,
                       textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)
    return pdf_path


# ─────────────────────────────────────────────
app = App(app_ui, server)
