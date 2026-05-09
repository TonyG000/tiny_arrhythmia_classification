================================================================================
ECG ARRHYTHMIA CLASSIFICATION — Tiny / Wearable-Ready
================================================================================

A 12-lead ECG arrhythmia classifier built around a CNN-Transformer architecture,
with explicit support for wearable deployment via random lead-masking
augmentation, hyperparameter tuning, and per-lead explainability analysis.

Final model accuracy on the CPSC 2018 test set: 79.6% (macro F1 = 0.772).
The same model retains 78.5% accuracy when only 3 leads (V1, II, V5 — the
clinical Holter montage) are active at inference, demonstrating wearable
feasibility.

--------------------------------------------------------------------------------
1. REPOSITORY LAYOUT
--------------------------------------------------------------------------------

    config.py                          Global configuration (paths, hyperparams,
                                       MODEL_NAME, lead-masking flag, etc.)
    requirements.txt                   Python dependencies (Python 3.12)

    data_processing/
        cpsc_signal_loader.py          Reads raw .mat ECG signals
        data_generator.py              Keras Sequence — applies preprocessing
                                       and the random lead-masking augmentation
                                       (each training sample randomly keeps 1,
                                       3, or 12 leads active)

    models/
        model/
            cnn_attentaion_bilstm_improve.py   Baseline CNN-BiLSTM
            cnn_attention_transformer.py       Main CNN-Transformer model
            cnn_bilstm_transformer_ensemble.py CNN-BiLSTM-Transformer ensemble
            cnn1d_resent_18.py                 1D-ResNet variant
        checkpoints/                   (gitignored — see "Model weights" below)

    train_custom_log_cpsc.py           Training entry-point. Reads MODEL_NAME
                                       from config.py, applies focal loss /
                                       weighted CE, saves best checkpoint.

    hyperparameter_tuning.py           Keras Tuner sweep over learning rate,
                                       focal-loss gamma, transformer heads,
                                       and FFN dim. Best trial: heads=8,
                                       ff_dim=512.

    generate_pr_curves.py              Reproduces per-class precision-recall
                                       curves for the final model.

    explainability/
        grad_cam.py                    Temporal Grad-CAM heatmaps on the deep
                                       conv layer (which timesteps matter)
        lead_importance.py             Per-lead attribution via input-gradient
                                       saliency + lead occlusion. Produces
                                       9-class x 12-lead importance matrices
                                       and the overall lead ranking.
        evaluate_3lead.py              Compares full 12-lead vs V1/V2/V3
                                       inference (the data-driven top-3).
        evaluate_montages.py           Multi-montage comparison: 12-lead,
                                       V1/V2/V3 (data-driven), V1/II/V5
                                       (clinical Holter), per-class consensus.
        comparative_analysis.py        Side-by-side eval of all checkpoints
                                       on the same test set.
        plot_comparative.py            Renders presentation-ready bar charts
                                       and heatmaps for the comparative table.

    demo.ipynb                         LIVE DEMO notebook. Loads the trained
                                       checkpoint and showcases predictions,
                                       Grad-CAM, lead importance, and the
                                       3-lead wearable scenario on real test
                                       samples. Used in the presentation demo.

    models/model_results/              All generated plots and tables (kept in
                                       git so the analysis is reproducible
                                       without re-running everything).

--------------------------------------------------------------------------------
2. SETUP — INSTALLATION
--------------------------------------------------------------------------------

Python 3.12 is required.

    # 1. Clone the repo
    git clone https://github.com/TonyG000/tiny_arrhythmia_classification.git
    cd tiny_arrhythmia_classification

    # 2. Create a virtual environment
    python3.12 -m venv .venv
    source .venv/bin/activate           # on Windows: .venv\Scripts\activate

    # 3. Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

    # 4. Install Jupyter (only needed to run demo.ipynb)
    pip install jupyter ipykernel

Tested on Linux x86_64 with TensorFlow 2.18 (CPU mode is fine for inference;
GPU is recommended for training).

--------------------------------------------------------------------------------
3. DATASET
--------------------------------------------------------------------------------

Dataset: CPSC 2018 — China Physiological Signal Challenge 2018
9 arrhythmia classes (AF, IAVB, LBBB, PAC, PVC, RBBB, SNR, STD, STE),
12-lead ECGs, ~6,800 recordings.

Permanent download links (the dataset itself is ~3 GB and is NOT included
in this repository):

    Official:   http://2018.icbeb.org/Challenge.html
    PhysioNet:  https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/

After downloading, point `DATASET_PATH` in config.py to the unzipped folder.

The pre-split train/val/test pickle files used here live at:
    data/train_data.pkl
    data/val_data.pkl
    data/test_data.pkl
These are produced by the project's preprocessing scripts. The test set
contains 736 samples and is what the demo notebook uses.

--------------------------------------------------------------------------------
4. MODEL WEIGHTS
--------------------------------------------------------------------------------

The final tuned + lead-masked checkpoint is required to run the demo. It is
NOT committed to git (the models/checkpoints/ folder is gitignored).

Permanent download link (GitHub Release):
    [PASTE LINK HERE AFTER UPLOADING TO GITHUB RELEASES]

After downloading, place the file at:
    models/checkpoints/CNNTransformerModel_lead_masked_tuned_best.h5

For reference, the other checkpoints used in the comparative analysis are:
    EnhancedCNNModel_best.h5                       (CNN-BiLSTM baseline)
    CNNTransformerModel_best_original.h5           (CNN-Transformer untuned)
    CNNTransformerModel_tuned_best.h5              (tuned, no lead masking)
    CNNTransformerModel_lead_masked_tuned_best.h5  (FINAL — used in the demo)
    EnsembleModel_best.h5                          (CNN-BiLSTM-Transformer ensemble)

--------------------------------------------------------------------------------
5. HOW TO RUN
--------------------------------------------------------------------------------

LIVE DEMO (the part the prof asked for)
    jupyter notebook demo.ipynb
    # Then run cells top-to-bottom. Loads in ~10 s, each demo cell
    # produces a plot of an ECG sample with predicted class + confidence.

TRAINING (only if you want to reproduce the model from scratch — takes
hours on CPU, faster on GPU)
    python train_custom_log_cpsc.py

HYPERPARAMETER TUNING (very long — produces the tuner artifacts that we
already used to pick the final architecture)
    python hyperparameter_tuning.py

EXPLAINABILITY / WEARABLE ANALYSIS (a few minutes each — they regenerate
the plots and numbers used in the slides)
    python -m explainability.lead_importance       # per-lead importance
    python -m explainability.evaluate_montages     # 3-lead comparisons
    python -m explainability.comparative_analysis  # all-models comparison
    python -m explainability.plot_comparative      # presentation plots
    python -m explainability.grad_cam              # temporal heatmaps

--------------------------------------------------------------------------------
6. KEY RESULTS
--------------------------------------------------------------------------------

Final model — CNN-Transformer (tuned + lead-masked) — on CPSC 2018 test set:

    Accuracy        0.7962
    Macro Recall    0.7973
    Macro F1        0.7721

Wearable montage comparison (same model, masked input at inference):

    Montage              Accuracy    Macro F1    Drop vs 12-lead
    --------------       --------    --------    ---------------
    12-lead (full)        0.7962      0.7721           --
    V1/II/V5 (Holter)     0.7853      0.7528         -1.1 pp
    V1/V2/V3 (data)       0.7242      0.6932         -7.2 pp

Full per-class numbers and presentation visuals are in
models/model_results/comparative/ and models/model_results/lead_importance/.

--------------------------------------------------------------------------------
7. CITATION
--------------------------------------------------------------------------------

Dataset:
    Liu F, Liu C, Zhao L, et al. "An Open Access Database for Evaluating
    the Algorithms of Electrocardiogram Rhythm and Morphology Abnormality
    Detection." Journal of Medical Imaging and Health Informatics.
    2018;8(7):1368-1373.

Author: Tony Gerges (TonyG000) — AUC, 2026.
