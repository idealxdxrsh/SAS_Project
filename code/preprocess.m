function [eeg_clean, epochs, epoch_labels] = preprocess(eeg, fs, ann_labels, ann_times)
%PREPROCESS  Filter EEG, remove artefacts, and segment into 30-second epochs
%
%  Inputs:
%    eeg        - raw EEG signal [N x 1]
%    fs         - sampling frequency (Hz)
%    ann_labels - cell array of stage strings
%    ann_times  - epoch onset times in seconds
%
%  Outputs:
%    eeg_clean   - filtered EEG [N x 1]
%    epochs      - [EPOCH_SAMPLES x num_epochs] matrix
%    epoch_labels- cell array of labels per epoch
%
%  Filters applied:
%    1. DC removal (mean subtraction)
%    2. FIR band-pass  0.3 – 35 Hz  (main signal band)
%    3. IIR notch      49 – 51 Hz   (powerline, Chebyshev Type-II)
%    4. Amplitude artefact rejection (|z-score| > 5 → NaN-zeroed)
% =========================================================================

    EPOCH_SEC    = 30;
    EPOCH_SAMP   = EPOCH_SEC * fs;

    fprintf('    Fs = %d Hz | Signal length = %.1f min\n', ...
            fs, numel(eeg)/(fs*60));

    %% ── 1. DC removal ────────────────────────────────────────────────────
    eeg = eeg - mean(eeg, 'omitnan');

    %% ── 2. FIR band-pass filter: 0.3 – 35 Hz ────────────────────────────
    fir_order = 3 * fix(fs / 0.3);   % ~10× longest period
    if mod(fir_order, 2) ~= 0, fir_order = fir_order + 1; end
    bp_fir = fir1(fir_order, [0.3 35] / (fs/2), 'bandpass');
    eeg = filtfilt(bp_fir, 1, eeg);
    fprintf('    FIR band-pass  (0.3–35 Hz, order %d) applied.\n', fir_order);

    %% ── 3. IIR notch: 50 Hz (Chebyshev Type-II) ─────────────────────────
    if fs > 110   % only if Nyquist > 55 Hz
        Wp = [49 51] / (fs/2);
        Ws = [48 52] / (fs/2);
        [n_ord, Wn] = cheb2ord(Wp, Ws, 1, 60);
        [b_notch, a_notch] = cheby2(n_ord, 60, Wn, 'stop');
        eeg = filtfilt(b_notch, a_notch, eeg);
        fprintf('    Chebyshev II notch (50 Hz, order %d) applied.\n', n_ord);
    end

    eeg_clean = eeg;

    %% ── 4. Epoch segmentation ────────────────────────────────────────────
    n_epochs_ann = numel(ann_labels);
    n_epochs_sig = floor(numel(eeg_clean) / EPOCH_SAMP);
    n_epochs     = min(n_epochs_ann, n_epochs_sig);

    fprintf('    Segmenting into %d epochs of %d s each.\n', n_epochs, EPOCH_SEC);

    epochs       = zeros(EPOCH_SAMP, n_epochs);
    epoch_labels = ann_labels(1:n_epochs);

    for e = 1:n_epochs
        idx_start = (e-1)*EPOCH_SAMP + 1;
        idx_end   = e * EPOCH_SAMP;
        seg = eeg_clean(idx_start:idx_end);

        % Amplitude artefact rejection: clip |z| > 5
        z = (seg - mean(seg)) / (std(seg) + eps);
        seg(abs(z) > 5) = 0;

        epochs(:, e) = seg;
    end

    %% ── Stage distribution summary ───────────────────────────────────────
    stages = {'W','N1','N2','N3','R'};
    fprintf('    Stage distribution:\n');
    for s = 1:numel(stages)
        cnt = sum(strcmp(epoch_labels, stages{s}));
        fprintf('      %s: %d epochs (%.1f min)\n', ...
                stages{s}, cnt, cnt*EPOCH_SEC/60);
    end
end
