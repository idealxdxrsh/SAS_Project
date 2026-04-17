function [spindles, kcomplexes] = detect_transients(eeg_clean, fs, stage_labels)
%DETECT_TRANSIENTS  Detect sleep spindles and K-complexes in EEG signal
%
%  Spindle detection:
%    - Sigma-band (11–16 Hz) band-pass filter
%    - Hilbert envelope smoothing (200 ms moving average)
%    - Threshold: mean + 2.5*std of envelope (computed in N2 epochs only)
%    - Duration gate: 0.5 – 3 s
%    - Only during N1/N2 epochs
%
%  K-complex detection:
%    - Low-pass filter < 4 Hz (isolate slow waves)
%    - Derivative-based peak detection (large negative deflection)
%    - Amplitude threshold: > 75 µV peak-to-peak within 1 s window
%    - Duration gate: 0.5 – 2 s
%    - Only during N2/N3 epochs
%
%  Inputs:
%    eeg_clean    - filtered EEG [N x 1]
%    fs           - sampling frequency
%    stage_labels - cell array of predicted stage labels (30-s epochs)
%
%  Outputs:
%    spindles    - struct array with fields: onset_s, offset_s, duration_s,
%                  peak_amp, freq_hz
%    kcomplexes  - struct array with fields: onset_s, offset_s, duration_s,
%                  neg_amp, pos_amp
% =========================================================================

    EPOCH_SEC  = 30;
    EPOCH_SAMP = EPOCH_SEC * fs;
    N          = numel(eeg_clean);

    %% ════════════════════════════════════════════════════════════════════
    %  SPINDLE DETECTION
    %% ════════════════════════════════════════════════════════════════════

    % ── Sigma band-pass: 11–16 Hz (FIR) ──────────────────────────────────
    ord_sp = round(6 * fs / 11);
    if mod(ord_sp,2)~=0, ord_sp=ord_sp+1; end
    b_sigma = fir1(ord_sp, [11 16]/(fs/2), 'bandpass');
    sig_sigma = filtfilt(b_sigma, 1, eeg_clean);

    % ── Hilbert envelope, smoothed 200 ms ────────────────────────────────
    env_sigma = abs(hilbert(sig_sigma));
    win_smooth = round(0.2 * fs);
    env_smooth = movmean(env_sigma, win_smooth);

    % ── Threshold from N2 epochs only ────────────────────────────────────
    n2_mask = build_stage_mask(stage_labels, {'N1','N2'}, EPOCH_SAMP, N);
    if sum(n2_mask) > 100
        ref_env = env_smooth(n2_mask);
    else
        ref_env = env_smooth;
    end
    thr_sp = mean(ref_env) + 2.5 * std(ref_env);

    % ── Threshold crossing detection ─────────────────────────────────────
    above     = env_smooth > thr_sp;
    spindle_mask = n2_mask & above;
    spindles  = extract_events(spindle_mask, eeg_clean, sig_sigma, fs, ...
                               0.5, 3.0, 'spindle');

    fprintf('    Spindle detection: threshold = %.2f µV | found %d spindles.\n', ...
            thr_sp, numel(spindles));

    %% ════════════════════════════════════════════════════════════════════
    %  K-COMPLEX DETECTION
    %% ════════════════════════════════════════════════════════════════════

    % ── Low-pass < 4 Hz to isolate slow waves ────────────────────────────
    ord_kc = round(6 * fs / 0.5);
    if mod(ord_kc,2)~=0, ord_kc=ord_kc+1; end
    b_lp = fir1(ord_kc, 3.5/(fs/2), 'low');
    sig_lp = filtfilt(b_lp, 1, eeg_clean);

    % ── Find large negative peaks (K-complex has sharp negative phase) ────
    n2n3_mask = build_stage_mask(stage_labels, {'N2','N3'}, EPOCH_SAMP, N);

    win_kc = round(1.0 * fs);   % 1-s sliding window
    kc_candidate = false(N, 1);

    % Amplitude threshold: 75 µV pp in 1-s window, only N2/N3
    % Auto-scale threshold to signal amplitude
    sig_range = prctile(sig_lp, 95) - prctile(sig_lp, 5);
    
    if sig_range < 1e-2     % signal likely in V
        amp_thr = 75e-6;
    elseif sig_range < 10   % signal likely in mV
        amp_thr = 75e-3;
    else                    % signal likely in µV
        amp_thr = 75;
    end

    % Rolling peak-to-peak
    for i = win_kc : N - win_kc
        if ~n2n3_mask(i), continue; end
        seg = sig_lp(i-win_kc+1 : i+win_kc);
        if (max(seg) - min(seg)) > amp_thr
            % Check sharp negative derivative
            dv = diff(seg);
            if min(dv) < -amp_thr/(2*fs)
                kc_candidate(i) = true;
            end
        end
    end

    % Cluster candidates into events
    kcomplexes = extract_events(kc_candidate, eeg_clean, sig_lp, fs, ...
                                0.5, 2.0, 'kcomplex');

    fprintf('    K-complex detection: amp_thr = %.2g | found %d K-complexes.\n', ...
            amp_thr, numel(kcomplexes));
end


%% ── Build boolean mask for specific stages ───────────────────────────────
function mask = build_stage_mask(labels, target_stages, epoch_samp, N)
    mask = false(N, 1);
    for e = 1:numel(labels)
        if ismember(labels{e}, target_stages)
            i0 = (e-1)*epoch_samp + 1;
            i1 = min(e*epoch_samp, N);
            mask(i0:i1) = true;
        end
    end
end


%% ── Extract event structs from binary mask ───────────────────────────────
function events = extract_events(mask, eeg, sig_filtered, fs, ...
                                  min_dur, max_dur, type)
    % Find contiguous runs
    d    = diff([0; mask(:); 0]);
    ons  = find(d ==  1);
    offs = find(d == -1) - 1;

    events = struct('onset_s',{}, 'offset_s',{}, 'duration_s',{}, ...
                    'peak_amp',{}, 'freq_hz',{});

    ev_idx = 1;
    for k = 1:numel(ons)
        dur = (offs(k) - ons(k) + 1) / fs;
        if dur < min_dur || dur > max_dur, continue; end

        seg = eeg(ons(k):offs(k));
        seg_f = sig_filtered(ons(k):offs(k));

        events(ev_idx).onset_s    = ons(k)  / fs;
        events(ev_idx).offset_s   = offs(k) / fs;
        events(ev_idx).duration_s = dur;
        events(ev_idx).peak_amp   = max(abs(seg));

        if strcmp(type, 'spindle')
            % Instantaneous frequency via zero-crossing rate
            zc = sum(abs(diff(sign(seg_f)))) / 2;
            events(ev_idx).freq_hz = zc / dur;
        else
            events(ev_idx).freq_hz = NaN;
        end

        ev_idx = ev_idx + 1;
    end
end