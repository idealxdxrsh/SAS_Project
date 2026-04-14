%% =========================================================================
%  AUTOMATED SLEEP EEG ANALYSIS — MAIN PIPELINE
%  Group 28 | BITS Pilani, Goa Campus
%
%  Pipeline Stages:
%    1. Data Loading        (load_edf.m)
%    2. Pre-processing      (preprocess.m)
%    3. Feature Extraction  (extract_features.m)
%    4. Sleep Stage Classif.(classify_stages.m)
%    5. Transient Detection (detect_transients.m)
%    6. Results & Export    (export_results.m)
%
%  Usage:
%    - Place .edf and annotation .txt files from PhysioNet Sleep-EDF
%      inside the  ../data/  folder.
%    - Run this script. Outputs go to  ../outputs/
%
%  Dependencies: Signal Processing Toolbox, Statistics & ML Toolbox
% =========================================================================

clc; clear; close all;

%% ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = fileparts(fileparts(mfilename('fullpath')));
CODE_DIR   = fullfile(BASE_DIR, 'code');
DATA_DIR   = fullfile(BASE_DIR, 'data');
OUT_DIR    = fullfile(BASE_DIR, 'outputs');
FIG_DIR    = fullfile(OUT_DIR,  'figures');
addpath(CODE_DIR);

% Create output folders
for d = {OUT_DIR, FIG_DIR}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fprintf('\n========================================================\n');
fprintf('  AUTOMATED SLEEP EEG ANALYSIS — GROUP 28\n');
fprintf('========================================================\n\n');

%% ── Discover EDF files ───────────────────────────────────────────────────
edf_files = dir(fullfile(DATA_DIR, '*PSG*.edf'));
if isempty(edf_files)
    edf_files = dir(fullfile(DATA_DIR, '*.edf'));
end

if isempty(edf_files)
    error(['No EDF files found in %s\n' ...
           'Download Sleep-EDF from: https://physionet.org/content/sleep-edfx/'], DATA_DIR);
end

fprintf('Found %d EDF file(s) to process.\n\n', numel(edf_files));

%% ── Per-recording pipeline ───────────────────────────────────────────────
all_results = struct();

for rec = 1:numel(edf_files)
    edf_path = fullfile(DATA_DIR, edf_files(rec).name);
    fprintf('[%d/%d] Processing: %s\n', rec, numel(edf_files), edf_files(rec).name);
    fprintf('--------------------------------------------------\n');

    % ── Stage 1: Load ────────────────────────────────────────────────────
    fprintf('  [1/6] Loading EDF data...\n');
    [eeg, fs, ann_labels, ann_times] = load_edf(edf_path, DATA_DIR);

    % ── Stage 2: Pre-process ─────────────────────────────────────────────
    fprintf('  [2/6] Pre-processing & filtering...\n');
    [eeg_clean, epochs, epoch_labels] = preprocess(eeg, fs, ann_labels, ann_times);

    % ── Stage 3: Feature Extraction ──────────────────────────────────────
    fprintf('  [3/6] Extracting features...\n');
    [features, feature_names] = extract_features(epochs, fs);

    % ── Stage 4: Stage Classification ────────────────────────────────────
    fprintf('  [4/6] Classifying sleep stages...\n');
    [pred_labels, conf_matrix, accuracy, kappa] = classify_stages(features, epoch_labels);

    % ── Stage 5: Transient Detection ─────────────────────────────────────
    fprintf('  [5/6] Detecting spindles and K-complexes...\n');
    [spindles, kcomplexes] = detect_transients(eeg_clean, fs, pred_labels);

    % ── Stage 6: Export ───────────────────────────────────────────────────
    fprintf('  [6/6] Exporting results & plots...\n');
    rec_name = erase(edf_files(rec).name, '.edf');
    export_results(rec_name, pred_labels, epoch_labels, conf_matrix, ...
                   accuracy, kappa, spindles, kcomplexes, ...
                   eeg_clean, fs, features, feature_names, OUT_DIR, FIG_DIR);

    % Store summary
    all_results(rec).name     = rec_name;
    all_results(rec).accuracy = accuracy;
    all_results(rec).kappa    = kappa;
    all_results(rec).n_spindles   = numel(spindles);
    all_results(rec).n_kcomplexes = numel(kcomplexes);

    fprintf('  Accuracy: %.1f%%  |  Kappa: %.3f  |  Spindles: %d  |  K-Cx: %d\n\n', ...
            accuracy*100, kappa, numel(spindles), numel(kcomplexes));
end

%% ── Summary CSV ──────────────────────────────────────────────────────────
fprintf('Writing summary CSV...\n');
T = struct2table(all_results);
writetable(T, fullfile(OUT_DIR, 'summary_all_recordings.csv'));

fprintf('\n========================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('  Outputs saved to: %s\n', OUT_DIR);
fprintf('========================================================\n');
