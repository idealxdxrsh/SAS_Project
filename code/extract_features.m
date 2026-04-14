function [features, feature_names] = extract_features(epochs, fs)
%EXTRACT_FEATURES  Compute spectral and time-freq features from EEG epochs
%
%  Inputs:
%    epochs  - [EPOCH_SAMP x num_epochs] filtered epoch matrix
%    fs      - sampling frequency (Hz)
%
%  Outputs:
%    features      - [num_epochs x num_features] feature matrix
%    feature_names - cell array of feature name strings
%
%  Features extracted per epoch:
%    Band power (absolute & relative): delta, theta, alpha, sigma, beta
%    Spectral edge frequency (SEF95)
%    Spectral entropy
%    Hjorth parameters: activity, mobility, complexity
%    Hilbert envelope: mean, std, peak
%    Wavelet sub-band energies (db4, 5 levels)
%    Total power
% =========================================================================

    [EPOCH_SAMP, n_epochs] = size(epochs);

    % ── Frequency band definitions (Hz) ──────────────────────────────────
    bands = struct( ...
        'delta', [0.5  4.0], ...
        'theta', [4.0  8.0], ...
        'alpha', [8.0 13.0], ...
        'sigma', [11.0 16.0], ...
        'beta',  [16.0 30.0]  ...
    );
    band_names = fieldnames(bands);
    n_bands    = numel(band_names);

    % ── Pre-allocate ──────────────────────────────────────────────────────
    % 5 abs + 5 rel + SEF + SpEnt + 3 Hjorth + 3 Hilbert + 5 wavelet + 1 total = 23
    N_FEAT   = n_bands*2 + 1 + 1 + 3 + 3 + n_bands + 1;
    features = zeros(n_epochs, N_FEAT);

    %% ── Build feature names ──────────────────────────────────────────────
    fnames = {};
    for b = 1:n_bands
        fnames{end+1} = ['bp_abs_' band_names{b}]; %#ok<AGROW>
    end
    for b = 1:n_bands
        fnames{end+1} = ['bp_rel_' band_names{b}]; %#ok<AGROW>
    end
    fnames = [fnames, {'sef95','spectral_entropy', ...
                        'hjorth_activity','hjorth_mobility','hjorth_complexity', ...
                        'hilbert_env_mean','hilbert_env_std','hilbert_env_peak'}];
    for b = 1:n_bands
        fnames{end+1} = ['wavelet_' band_names{b}]; %#ok<AGROW>
    end
    fnames{end+1} = 'total_power';
    feature_names = fnames;

    %% ── Welch PSD setup ──────────────────────────────────────────────────
    win_len  = 2 * fs;          % 2-second Hann window
    n_fft    = 2^nextpow2(win_len * 2);
    overlap  = round(win_len * 0.5);
    freq_res = fs / n_fft;
    freqs    = (0 : n_fft/2) * freq_res;

    %% ── Process each epoch ───────────────────────────────────────────────
    for e = 1:n_epochs
        x = epochs(:, e);

        % ── Welch PSD ────────────────────────────────────────────────────
        [pxx, ~] = pwelch(x, hann(win_len), overlap, n_fft, fs);
        total_pow = sum(pxx) * freq_res;

        col = 1;

        % Absolute band powers
        for b = 1:n_bands
            f_range = bands.(band_names{b});
            idx = freqs >= f_range(1) & freqs <= f_range(2);
            features(e, col) = sum(pxx(idx)) * freq_res;
            col = col + 1;
        end

        % Relative band powers
        for b = 1:n_bands
            f_range = bands.(band_names{b});
            idx = freqs >= f_range(1) & freqs <= f_range(2);
            features(e, col) = sum(pxx(idx)) * freq_res / (total_pow + eps);
            col = col + 1;
        end

        % Spectral edge frequency (95%)
        cum_pow = cumsum(pxx) * freq_res;
        sef_idx = find(cum_pow >= 0.95 * total_pow, 1);
        features(e, col) = freqs(min(sef_idx, numel(freqs)));
        col = col + 1;

        % Spectral entropy (normalised)
        p_norm = pxx / (sum(pxx) + eps);
        p_norm(p_norm <= 0) = eps;
        features(e, col) = -sum(p_norm .* log2(p_norm));
        col = col + 1;

        % ── Hjorth parameters ─────────────────────────────────────────────
        dx  = diff(x);
        ddx = diff(dx);
        act = var(x);
        mob = sqrt(var(dx) / (act + eps));
        cmp = sqrt(var(ddx) / (var(dx) + eps)) / (mob + eps);
        features(e, col)   = act;
        features(e, col+1) = mob;
        features(e, col+2) = cmp;
        col = col + 3;

        % ── Hilbert envelope ──────────────────────────────────────────────
        env = abs(hilbert(x));
        features(e, col)   = mean(env);
        features(e, col+1) = std(env);
        features(e, col+2) = max(env);
        col = col + 3;

        % ── Wavelet sub-band energies (db4, 5-level DWT) ─────────────────
        wav_energies = wavelet_band_energies(x, fs);
        features(e, col:col+n_bands-1) = wav_energies;
        col = col + n_bands;

        % ── Total power ───────────────────────────────────────────────────
        features(e, col) = total_pow;
    end

    fprintf('    Features extracted: %d per epoch (%d epochs total).\n', ...
            N_FEAT, n_epochs);
end


%% ── Wavelet band energy helper ───────────────────────────────────────────
function energies = wavelet_band_energies(x, fs)
%  Decompose with db4 to 5 levels.
%  Level-band mapping (approximate, for fs=100 Hz):
%    d1: 25–50 Hz  (beta+)
%    d2: 12.5–25 Hz (beta)
%    d3: 6.25–12.5 Hz (alpha/sigma)
%    d4: 3.12–6.25 Hz (theta)
%    d5: 1.56–3.12 Hz (delta high)
%    a5: 0–1.56 Hz    (delta low)
%
%  We sum d4+d5→delta, d4→theta, d3→alpha, d3→sigma, d2→beta
%  (simplified mapping; actual boundaries scale with fs)

    n_levels = 5;
    [c, l] = wavedec(x, n_levels, 'db4');

    energies = zeros(1, 5);
    % d5 → delta, d4 → theta, d3 → alpha/sigma, d2 → beta, d1 → high
    level_map = [5, 4, 3, 2, 1];   % delta, theta, alpha, sigma, beta
    for k = 1:5
        d = detcoef(c, l, level_map(k));
        energies(k) = sum(d.^2) / numel(d);
    end
end
