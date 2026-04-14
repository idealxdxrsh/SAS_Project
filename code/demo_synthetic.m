%% =========================================================================
%  DEMO_SYNTHETIC.m — Run full pipeline on synthetic EEG (no data needed)
%
%  This script generates realistic synthetic EEG signals for each sleep
%  stage and exercises every module end-to-end.
%  Use this to verify the pipeline runs correctly before using real EDF files.
%
%  Usage: Run this script from the /code/ folder.
% =========================================================================

clc; clear; close all;
addpath(fileparts(mfilename('fullpath')));

BASE_DIR = fileparts(fileparts(mfilename('fullpath')));
OUT_DIR  = fullfile(BASE_DIR, 'outputs');
FIG_DIR  = fullfile(OUT_DIR,  'figures');
for d = {OUT_DIR, FIG_DIR}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

fprintf('========================================================\n');
fprintf('  SYNTHETIC EEG DEMO — GROUP 28\n');
fprintf('========================================================\n\n');

%% ── Parameters ───────────────────────────────────────────────────────────
fs         = 100;      % Hz
EPOCH_SEC  = 30;
EPOCH_SAMP = EPOCH_SEC * fs;

% Simulate a 6-hour night: stage sequence
stage_seq = {'W','W','N1','N2','N2','N3','N3','N2','R','R', ...
             'N2','N3','N3','N2','R','R','N2','N2','N3','N2', ...
             'R','R','R','N2','N2','N1','W','W','N2','N2', ...
             'R','R','N2','N1','W','W','N2','N3','N2','R', ...
             'R','N2','N1','W'};
n_epochs = numel(stage_seq);
ann_times = (0:n_epochs-1)' * EPOCH_SEC;

fprintf('Simulating %d epochs (%.1f min) of EEG...\n', n_epochs, n_epochs*EPOCH_SEC/60);

%% ── Generate synthetic EEG ───────────────────────────────────────────────
rng(42);
eeg_raw = zeros(n_epochs * EPOCH_SAMP, 1);

% Band-power profiles per stage (µV²/Hz): [delta theta alpha sigma beta]
stage_profiles = struct( ...
    'W',  [0.1  0.2  1.5  0.2  0.3], ...
    'N1', [0.5  0.8  0.4  0.3  0.2], ...
    'N2', [1.5  0.5  0.1  0.8  0.1], ...
    'N3', [4.0  0.3  0.05 0.1  0.05], ...
    'R',  [0.3  0.6  0.2  0.2  0.4]  ...
);
band_freqs = [2, 6, 10, 13.5, 23];   % representative freq per band
noise_amp  = 5;                        % µV white noise

for e = 1:n_epochs
    profile = stage_profiles.(stage_seq{e});
    t_seg   = (0:EPOCH_SAMP-1)' / fs;
    seg     = noise_amp * randn(EPOCH_SAMP, 1);   % base noise
    for b = 1:5
        amp = sqrt(2 * profile(b));
        phi = 2*pi*rand;
        seg = seg + amp * sin(2*pi*band_freqs(b)*t_seg + phi);
    end
    % Add spindles in N2 epochs
    if strcmp(stage_seq{e}, 'N2') && rand < 0.5
        sp_start = round(rand * (EPOCH_SAMP - 2*fs)) + 1;
        sp_len   = round((0.5 + rand*1.0) * fs);
        t_sp     = (0:sp_len-1)'/fs;
        sp_env   = sin(pi * t_sp / (sp_len/fs));
        seg(sp_start:sp_start+sp_len-1) = seg(sp_start:sp_start+sp_len-1) + ...
                                           30 * sp_env .* sin(2*pi*13.5*t_sp);
    end
    % Add K-complexes in N2/N3
    if ismember(stage_seq{e}, {'N2','N3'}) && rand < 0.3
        kc_pos = round(rand * (EPOCH_SAMP - round(1.5*fs))) + 1;
        kc_len = round(1.0 * fs);
        t_kc   = (0:kc_len-1)'/fs;
        kc_wave = 100 * sin(pi*t_kc/1.0) .* sin(2*pi*1.5*t_kc);
        seg(kc_pos:kc_pos+kc_len-1) = seg(kc_pos:kc_pos+kc_len-1) + kc_wave;
    end

    i0 = (e-1)*EPOCH_SAMP + 1;
    eeg_raw(i0:i0+EPOCH_SAMP-1) = seg;
end

fprintf('  Synthetic EEG generated: %.1f seconds, fs=%d Hz\n\n', ...
        numel(eeg_raw)/fs, fs);

%% ── Run pipeline ─────────────────────────────────────────────────────────
fprintf('[2/6] Pre-processing...\n');
[eeg_clean, epochs, epoch_labels] = preprocess(eeg_raw, fs, stage_seq', ann_times);

fprintf('\n[3/6] Feature extraction...\n');
[features, feat_names] = extract_features(epochs, fs);

fprintf('\n[4/6] Classification...\n');
[pred_labels, conf_matrix, accuracy, kappa] = classify_stages(features, epoch_labels);

fprintf('\n[5/6] Transient detection...\n');
[spindles, kcomplexes] = detect_transients(eeg_clean, fs, pred_labels);

fprintf('\n[6/6] Exporting results...\n');
export_results('DEMO_synthetic', pred_labels, epoch_labels, conf_matrix, ...
               accuracy, kappa, spindles, kcomplexes, ...
               eeg_clean, fs, features, feat_names, OUT_DIR, FIG_DIR);

%% ── Summary ──────────────────────────────────────────────────────────────
fprintf('\n========================================================\n');
fprintf('  DEMO COMPLETE\n');
fprintf('  Accuracy : %.1f%%\n', accuracy*100);
fprintf('  Kappa    : %.3f\n',   kappa);
fprintf('  Spindles : %d detected\n', numel(spindles));
fprintf('  K-Cx     : %d detected\n', numel(kcomplexes));
fprintf('  Outputs  : %s\n', OUT_DIR);
fprintf('========================================================\n');
