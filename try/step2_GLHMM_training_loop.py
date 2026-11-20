import os
import glob
import gc
import numpy as np

from glhmm import glhmm, io

# =========================
# SETTINGS
# =========================

data_root = r"C:\Users\pipo\Datasets-Docs\Multimodal Dataset for Mixed Emotion Recognition\preprocessed_GLHMM"
K = 4 # Define state number for GLHMM
model_folder_name = f"GLHMM_K{K}_noPCA"

train_options = {
    "cyc": 200,
    "tol": 1e-4,
    "initrep": 3,
    "serial": True,
    "deactivate_states": False,
    "verbose": False,
    "gpu_acceleration": 0
}

# =========================
# FIND SUBJECT FILES
# =========================

subject_files = sorted(glob.glob(
    os.path.join(data_root, "*", "DMER_GLHMM_features.npz")
))

print("Total subjects:", len(subject_files))

# =========================
# SUBJECT LOOP
# =========================

for i, file_path in enumerate(subject_files):

    subject_dir = os.path.dirname(file_path)
    subject_id = os.path.basename(subject_dir)

    save_dir = os.path.join(
        subject_dir,
        model_folder_name
    )

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "GLHMM_model.pkl")
    result_path = os.path.join(save_dir, "GLHMM_results.npz")

    # Skip completed subjects
    if os.path.exists(model_path) and os.path.exists(result_path):
        print(f"Subject {subject_id}: Already trained. Skipping.")
        continue

    # Initialize variables for memory cleanup
    X = Y = indices = None
    hmm = Gamma = Xi = FE = None

    try:

        print(f"\n===== Subject {subject_id} ({i+1}/{len(subject_files)}) =====")

        # =========================
        # LOAD DATA
        # =========================

        with np.load(file_path, allow_pickle=False) as data:

            X = data["X"].astype(np.float64)
            Y = data["Y"].astype(np.float64)
            indices = data["indices"].astype(int)

        print("EEG:", Y.shape)
        print("Physiology:", X.shape)
        print("Trials:", indices.shape)

        # Validate data
        assert X.shape[0] == Y.shape[0]
        assert indices.shape[1] == 2
        assert indices[0, 0] == 0
        assert indices[-1, 1] == Y.shape[0]
        assert np.all(indices[1:, 0] == indices[:-1, 1])
        assert np.isfinite(X).all()
        assert np.isfinite(Y).all()

        # =========================
        # INITIALIZE GLHMM
        # =========================

        np.random.seed(123 + i)

        hmm = glhmm.glhmm(
            K=K,
            covtype="diag",
            model_mean="state",
            model_beta="state",
            dirichlet_diag=10
        )

        # =========================
        # TRAIN GLHMM
        # =========================

        Gamma, Xi, FE = hmm.train(
            X=X,
            Y=Y,
            indices=indices,
            options=train_options
        )

        print("Gamma:", Gamma.shape)
        print("Xi:", Xi.shape)
        print("Final FE:", np.asarray(FE).ravel()[-1])

        # =========================
        # SAVE RESULTS
        # =========================

        io.save_hmm(
            hmm,
            "GLHMM_model.pkl",
            directory=save_dir
        )

        np.savez_compressed(
            result_path,
            Gamma=Gamma,
            Xi=Xi,
            FE=FE,
            indices=indices
        )

        print(f"Subject {subject_id}: Training completed.")

    except Exception as e:

        print(f"Subject {subject_id}: ERROR - {e}")

    finally:

        # =========================
        # MEMORY CLEANUP
        # =========================

        del X, Y, indices
        del hmm, Gamma, Xi, FE

        gc.collect()

        print(f"Subject {subject_id}: Memory released.")

print("\n===== ALL SUBJECTS FINISHED =====")