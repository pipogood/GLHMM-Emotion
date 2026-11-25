import os
import glob
import gc
import numpy as np

from glhmm import glhmm, io

# =========================
# PATHS
# =========================

data_root = r"C:\Users\pipo\Datasets-Docs\Multimodal Dataset for Mixed Emotion Recognition\preprocessed_GLHMM"

save_dir = os.path.join(data_root, "GLHMM_Group_K4_noPCA_05sec_fullCov")  ## change this for output
os.makedirs(save_dir, exist_ok=True)

subject_files = sorted(
    glob.glob(
        os.path.join(
            data_root,
            "*",
            "DMER_GLHMM_features_05sec.npz" ## change this for input
        )
    )
)

print("Subjects found:", len(subject_files))

# =========================
# LOAD ALL SUBJECTS
# =========================

X, Y, indices, _ = io.load_files(subject_files)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Indices shape:", indices.shape)

assert X.shape[1] == 5
assert Y.shape[1] == 90
assert indices.shape[0] == len(subject_files) * 32
assert indices[-1, 1] == len(Y)

# =========================
# CREATE GROUP MODEL
# =========================

K = 4

hmm = glhmm.glhmm(
    K=K,
    # covtype="diag",
    covtype="full",
    model_mean="state",
    model_beta="state",
    dirichlet_diag=5
)

# =========================
# TRAINING OPTIONS
# =========================

train_options = {
    "cyc": 300,
    "tol": 1e-4,
    "initrep": 3,
    "serial": True,
    "deactivate_states": False,
    "verbose": True,
    "gpu_acceleration": 0
}

# =========================
# STANDARD GROUP TRAINING
# =========================

Gamma_train, Xi_train, FE = hmm.train(
    X=X,
    Y=Y,
    indices=indices,
    options=train_options
)

# =========================
# SAVE TRAINED MODEL
# =========================

io.save_hmm(
    hmm,
    "GLHMM_Group_model.pkl",
    directory=save_dir
)

np.savez_compressed(
    os.path.join(save_dir, "GLHMM_Group_training_results.npz"),
    Gamma=Gamma_train,
    Xi=Xi_train,
    FE=FE,
    indices=indices
)

print("Group training completed!")

# =========================
# MEMORY CLEANUP
# =========================

del X, Y, Gamma_train, Xi_train, FE, hmm
gc.collect()