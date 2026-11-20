
import scipy.io as sio
import numpy as np
import os
import pandas as pd
from scipy.signal import welch


path = r"C:\Users\pipo\Datasets-Docs\Multimodal Dataset for Mixed Emotion Recognition\Aligned_data_001"
files = os.listdir(path)

save_path = r"C:\Users\pipo\Datasets-Docs\Multimodal Dataset for Mixed Emotion Recognition\preprocessed_GLHMM"
save_name = "DMER_GLHMM_features.npz"  # Don't forget to change the name of the file if you change the features or preprocessing steps

# =========================
# SETTINGS
# =========================
eeg_window = 4  # seconds
gsr_window = 5
ppg_window = 8

step = 1  # second

freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 50)
}

for i in files:
    # EEG, GSR, and PPG data loading
    if i == "2" or i == "35":
        continue  # Skip subjects 2 and 35 due to missing data
    else:
        print(f"################ Processing subject {i}...")
        subject_id = i
        path_data = f"C:\\Users\\pipo\\Datasets-Docs\\Multimodal Dataset for Mixed Emotion Recognition\\Aligned_data_001\\{subject_id}"
        data = sio.loadmat(path_data + r"\datas.mat")

        # Behavioral data loading
        path_behave_VAD = f"C:\\Users\\pipo\\Datasets-Docs\\Multimodal Dataset for Mixed Emotion Recognition\\Raw_data_001\\{subject_id}\\{subject_id}\\Arousal_Valence.csv"
        path_behave_Emotion = f"C:\\Users\\pipo\\Datasets-Docs\\Multimodal Dataset for Mixed Emotion Recognition\\Raw_data_001\\{subject_id}\\{subject_id}\\Emotions.csv"
        path_behave_PANAS = f"C:\\Users\\pipo\\Datasets-Docs\\Multimodal Dataset for Mixed Emotion Recognition\\Raw_data_001\\{subject_id}\\{subject_id}\\Panas.csv"

        eeg = data["eeg_datas"]
        gsr = data["gsr_datas"]
        ppg = data["ppg_datas"]

        fs_eeg = 300
        fs_gsr = 4
        fs_ppg = 100

        eeg_trials = []
        gsr_trials = []
        ppg_trials = []

        behave_VAD = pd.read_csv(path_behave_VAD, header=None).to_numpy()
        behave_Emotion = pd.read_csv(path_behave_Emotion, header=None).to_numpy()
        behave_PANAS = pd.read_csv(path_behave_PANAS, header=None).to_numpy()

        behave_VAD_trials = [] # Valence, Arousal, Dominance values for each trial
        behave_Emotion_trials = [] # amusement and disgust values for each trial
        behave_PANAS_trials = [] # PANAS values for each trial

        for vid in range(32):
            # EEG: 18 channels
            eeg_trial = eeg[:18, eeg[-2, :] == vid]

            # GSR: 1 channel
            gsr_trial = gsr[:1, gsr[-2, :] == vid]

            # PPG: 1 channel
            ppg_trial = ppg[:1, ppg[-2, :] == vid]
            
            # Map 1st column of behavioral data to the corresponding video ID
            VAD_trial = behave_VAD[behave_VAD[:, 0] == vid, 1:]
            Emotion_trial = behave_Emotion[behave_Emotion[:, 0] == vid, 1:]
            PANAS_trial = behave_PANAS[behave_PANAS[:, 0] == vid, 1:]

            eeg_trials.append(eeg_trial)
            gsr_trials.append(gsr_trial)
            ppg_trials.append(ppg_trial)
            
            behave_VAD_trials.append(VAD_trial)
            behave_Emotion_trials.append(Emotion_trial)
            behave_PANAS_trials.append(PANAS_trial)
            

        # =========================
        # FEATURE STORAGE
        # =========================
        eeg_features = []
        gsr_features = []
        ppg_features = []

        trial_lengths = []
        feature_times = []
        trial_ids = []

        # =========================
        # EXTRACT FEATURES
        # =========================
        for vid in range(32):

            eeg_trial = eeg_trials[vid]
            gsr_trial = gsr_trials[vid]
            ppg_trial = ppg_trials[vid]

            # Common duration for this trial
            duration = int(min(
                eeg_trial.shape[1] / fs_eeg,
                gsr_trial.shape[1] / fs_gsr,
                ppg_trial.shape[1] / fs_ppg
            ))

            # Align the ends of the three recordings
            eeg_trial = eeg_trial[:, -duration * fs_eeg:]
            gsr_trial = gsr_trial[:, -duration * fs_gsr:]
            ppg_trial = ppg_trial[:, -duration * fs_ppg:]

            eeg_fea_trial = []
            gsr_fea_trial = []
            ppg_fea_trial = []

            # First valid time point = 8 seconds
            
            for t in np.arange(
                max(eeg_window, gsr_window, ppg_window),
                duration + step / 2, step):

                # =========================
                # EEG FEATURES
                # =========================

                eeg_seg = eeg_trial[
                    :,
                    int(round((t - eeg_window) * fs_eeg)):
                    int(round(t * fs_eeg))
                ]

                freqs, psd = welch(
                    eeg_seg,
                    fs=fs_eeg,
                    nperseg=eeg_seg.shape[1],
                    axis=1
                )

                eeg_fea = []

                for band, (low, high) in freq_bands.items():

                    mask = (freqs >= low) & (freqs < high)

                    band_power = (
                        psd[:, mask].sum(axis=1)
                        * (freqs[1] - freqs[0])
                    )

                    log_power = np.log10(band_power + 1e-18)

                    eeg_fea.extend(log_power)

                eeg_fea_trial.append(eeg_fea)

                # =========================
                # GSR FEATURES
                # =========================
                
                
                gsr_seg = gsr_trial[
                    0,
                    int(round((t - gsr_window) * fs_gsr)):
                    int(round(t * fs_gsr))
                ]

                gsr_mean = np.mean(gsr_seg)
                gsr_std = np.std(gsr_seg)

                gsr_slope = (
                    (gsr_seg[-1] - gsr_seg[0])
                    / ((len(gsr_seg) - 1) / fs_gsr)
                )

                gsr_fea_trial.append([
                    gsr_mean,
                    gsr_std,
                    gsr_slope
                ])

                # =========================
                # PPG FEATURES
                # =========================

                ppg_seg = ppg_trial[
                    0,
                    int(round((t - ppg_window) * fs_ppg)):
                    int(round(t * fs_ppg))
                ]

                ppg_std = np.std(ppg_seg)

                # Estimate pulse rate from spectral peak
                f_ppg, psd_ppg = welch(
                    ppg_seg,
                    fs=fs_ppg,
                    nperseg=len(ppg_seg),
                    nfft=4096
                )

                # Approximate 42–180 beats/min
                mask = (f_ppg >= 0.7) & (f_ppg <= 3.0)

                peak_freq = f_ppg[mask][
                    np.argmax(psd_ppg[mask])
                ]

                pulse_rate = peak_freq * 60

                ppg_fea_trial.append([
                    pulse_rate,
                    ppg_std
                ])

            # =========================
            # STORE TRIAL FEATURES
            # =========================
            eeg_features.append(np.array(eeg_fea_trial))
            gsr_features.append(np.array(gsr_fea_trial))
            ppg_features.append(np.array(ppg_fea_trial))

            trial_lengths.append(len(eeg_fea_trial))


            feature_times.append(
                np.arange(
                    max(eeg_window, gsr_window, ppg_window),
                    duration + step / 2,
                    step
                )
            )

            trial_ids.append(
                np.full(len(eeg_fea_trial), vid)
            )

        print("Feature extraction completed!")
            
        EEG = np.concatenate(eeg_features, axis=0)
        GSR = np.concatenate(gsr_features, axis=0)
        PPG = np.concatenate(ppg_features, axis=0)

        # GLHMM inputs
        Y = EEG
        X = np.concatenate([GSR, PPG], axis=1)

        # Keep video IDs
        video_ids_features = np.concatenate(trial_ids)
        
        
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Check for invalid values
        assert np.isfinite(Y).all(), "EEG contains NaN or Inf"
        assert np.isfinite(X).all(), "GSR/PPG contains NaN or Inf"

        # Initialize scalers
        scaler_eeg = StandardScaler()
        scaler_physio = StandardScaler()

        # Standardize each feature across all trials
        Y_std = scaler_eeg.fit_transform(Y)
        X_std = scaler_physio.fit_transform(X)
        
        
        # indices set fot GLHMM
        end_indices = np.cumsum(trial_lengths)
        start_indices = np.concatenate([[0], end_indices[:-1]])

        indices = np.column_stack([
            start_indices,
            end_indices
        ])

        print("Indices:", indices.shape)
        print(indices)
        
        # Labeling for each trial
        trial_labels = np.array(
            [0]*8 + [1]*8 + [2]*16
        )

        # Label for every feature time point
        emotion_labels = np.concatenate([
            np.full(trial_lengths[i], trial_labels[i])
            for i in range(32)
        ])
        
        save_path_subject = os.path.join(save_path, f"subject_{subject_id}")
        ## Making folder to save data
        if not os.path.exists(save_path_subject):
            os.makedirs(save_path_subject)

        #### saving data
        np.savez_compressed(
            os.path.join(save_path_subject, save_name),
            X=X_std,
            Y=Y_std,
            indices=indices,
            emotion_labels=emotion_labels,
            trial_labels=trial_labels,
            video_ids=video_ids_features,
            VAD=behave_VAD_trials,
            Emotion=behave_Emotion_trials,
            PANAS=behave_PANAS_trials
        )

