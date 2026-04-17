function [eeg, fs, ann_labels, ann_times] = load_edf(edf_path, data_dir)
%LOAD_EDF  Load EEG signal and sleep stage annotations from PhysioNet Sleep-EDF
%
%  Inputs:
%    edf_path  - full path to the PSG .edf file
%    data_dir  - folder containing annotation .txt/.edf files
%
%  Outputs:
%    eeg        - [N x 1] EEG signal (Fpz-Cz channel preferred)
%    fs         - sampling frequency (Hz)
%    ann_labels - [M x 1] cell array of stage strings {'W','N1','N2','N3','R'}
%    ann_times  - [M x 1] onset times in seconds
%
%  Note: Uses MATLAB's edfread() (R2020b+). For older MATLAB, falls back
%        to a built-in minimal EDF reader (read_edf_minimal).
% =========================================================================

    %% ── Read EDF signal ──────────────────────────────────────────────────
    try
        % Modern R2020b+ native edfread via edfinfo
        info = edfinfo(edf_path);
        signal_labels = info.SignalLabels;

        % Prefer Fpz-Cz, fallback to EEG, then first channel
        chan_idx = find(contains(signal_labels, 'Fpz-Cz', 'IgnoreCase', true), 1);
        if isempty(chan_idx)
            chan_idx = find(contains(signal_labels, 'EEG', 'IgnoreCase', true), 1);
        end
        if isempty(chan_idx), chan_idx = 1; end
        
        target_sig = signal_labels{chan_idx};

        % Read just the target signal to save memory
        tt = edfread(edf_path, 'SelectedSignals', target_sig);
        
        % edfread timetables convert spaces/hyphens to underscores, 
        % so we grab the first column dynamically rather than by exact name
        vars = tt.Properties.VariableNames;
        sig_data = tt.(vars{1});
        
        % Concatenate records into a single continuous vector
        if iscell(sig_data)
            eeg = vertcat(sig_data{:});
        else
            eeg = sig_data;
        end
        eeg = eeg(:);   
        
        % Calculate Sampling Frequency
        fs = round(info.NumSamples(chan_idx) / seconds(info.DataRecordDuration));

        fprintf('    Channel: %s  |  Fs: %d Hz  |  Duration: %.1f min\n', ...
                target_sig, fs, numel(eeg)/(fs*60));

    catch ME
        % Fallback minimal reader
        warning('Native edfread failed. Using minimal EDF reader. Error: %s', ME.message);
        [eeg, fs] = read_edf_minimal(edf_path);
    end

    %% ── Find annotation file ─────────────────────────────────────────────
    % Sleep-EDF naming: SC4001E0-PSG.edf → SC4001EC-Hypnogram.edf or .txt
    [~, base, ~] = fileparts(edf_path);
    base_id = base(1:6);   % e.g. 'SC4001'

    % Try hypnogram EDF first, then txt
    ann_edf = dir(fullfile(data_dir, [base_id '*Hypnogram*.edf']));
    ann_txt = dir(fullfile(data_dir, [base_id '*annotation*.txt']));
    ann_txt2= dir(fullfile(data_dir, [base_id '*hypnogram*.txt']));

    if ~isempty(ann_edf)
        [ann_labels, ann_times] = parse_hypnogram_edf(fullfile(data_dir, ann_edf(1).name));
    elseif ~isempty(ann_txt)
        [ann_labels, ann_times] = parse_annotation_txt(fullfile(data_dir, ann_txt(1).name));
    elseif ~isempty(ann_txt2)
        [ann_labels, ann_times] = parse_annotation_txt(fullfile(data_dir, ann_txt2(1).name));
    else
        warning('No annotation file found for %s. Using dummy labels.', base);
        n_epochs = floor(numel(eeg) / (30 * fs));
        ann_labels = repmat({'N2'}, n_epochs, 1);
        ann_times  = (0:n_epochs-1)' * 30;
    end

    fprintf('    Annotations: %d epochs loaded.\n', numel(ann_labels));
end


%% ── Helper: parse Hypnogram EDF ─────────────────────────────────────────
function [labels, times] = parse_hypnogram_edf(hyp_path)
    try
        % Modern R2020b+ annotation parsing via edfinfo
        info = edfinfo(hyp_path);
        
        % Extract raw onsets, durations, and labels in seconds
        raw_onsets = seconds(info.Annotations.Onset);
        raw_durations = seconds(info.Annotations.Duration);
        raw_labels = info.Annotations.Annotations;
        
        % Ensure labels are a cell array of strings
        if isstring(raw_labels)
            raw_labels = cellstr(raw_labels);
        end
        
        EPOCH_SEC = 30;
        labels = {};
        times = [];
        
        % Expand duration blocks into 30-second epochs
        for i = 1:numel(raw_labels)
            % Skip metadata/movement events that aren't sleep stages
            if isempty(raw_labels{i}) || contains(raw_labels{i}, 'Recording', 'IgnoreCase', true)
                continue;
            end
            
            % If a stage event is extremely short (e.g., < 15s), skip it.
            % Otherwise, calculate how many 30s epochs fit inside.
            n_epochs = round(raw_durations(i) / EPOCH_SEC);
            
            if n_epochs > 0
                % Generate an array of onsets for this block
                block_times = raw_onsets(i) + (0:n_epochs-1)' * EPOCH_SEC;
                
                % Duplicate the label for every epoch in the block
                block_labels = repmat(raw_labels(i), n_epochs, 1);
                
                times = [times; block_times]; %#ok<AGROW>
                labels = [labels; block_labels]; %#ok<AGROW>
            end
        end
        
    catch ME
        warning('Failed to parse Hypnogram EDF: %s', ME.message);
        [labels, times] = deal({}, []);
    end

    % Map Sleep-EDF stage strings → standard labels
    labels = map_stage_labels(labels);
end


%% ── Helper: parse plain-text annotation file ─────────────────────────────
function [labels, times] = parse_annotation_txt(txt_path)
    fid = fopen(txt_path, 'r');
    labels = {};
    times  = [];
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), break; end
        % Common format: "onset_s  duration_s  stage_string"
        parts = strsplit(strtrim(line));
        if numel(parts) >= 3
            t = str2double(parts{1});
            if ~isnan(t)
                times(end+1,1)  = t; %#ok<AGROW>
                labels{end+1,1} = parts{end}; %#ok<AGROW>
            end
        end
    end
    fclose(fid);
    labels = map_stage_labels(labels);
end


%% ── Helper: normalise stage string labels ────────────────────────────────
function labels = map_stage_labels(raw)
    labels = cell(size(raw));
    for i = 1:numel(raw)
        s = upper(strtrim(raw{i}));
        if contains(s, {'WAKE','W','STAGE W','0'})
            labels{i} = 'W';
        elseif contains(s, {'N1','STAGE 1','1','NREM1'})
            labels{i} = 'N1';
        elseif contains(s, {'N2','STAGE 2','2','NREM2'})
            labels{i} = 'N2';
        elseif contains(s, {'N3','N4','STAGE 3','STAGE 4','3','4','SWS'})
            labels{i} = 'N3';
        elseif contains(s, {'REM','R','STAGE R','5'})
            labels{i} = 'R';
        else
            labels{i} = 'N2';   % default unknown → N2
        end
    end
end


%% ── Helper: decode EDF+ TAL byte stream ─────────────────────────────────
function [labels, times] = decode_tal(raw)
    % Minimal TAL decoder — splits on null bytes
    txt = char(raw(:)');
    segs = strsplit(txt, char(0));
    labels = {};
    times  = [];
    for i = 1:numel(segs)
        seg = strtrim(segs{i});
        if isempty(seg), continue; end
        parts = strsplit(seg, char(20));   % ASCII 20 = record sep
        if numel(parts) >= 2
            t = str2double(parts{1});
            if ~isnan(t)
                times(end+1,1)  = t;
                labels{end+1,1} = strtrim(parts{end});
            end
        end
    end
end


%% ── Fallback: minimal EDF binary reader ─────────────────────────────────
function [signal, fs] = read_edf_minimal(path)
    fid = fopen(path, 'r', 'ieee-le');
    % Skip 256-byte global header
    fseek(fid, 252, 'bof');
    ns = str2double(fread(fid, 4, '*char')');   % num signals
    fseek(fid, 256, 'bof');
    % Read per-signal headers
    labels    = cellstr(reshape(fread(fid, 16*ns, '*char'), 16, ns)');
    fseek(fid, 256 + 216*ns, 'bof');
    fs_cell   = cellstr(reshape(fread(fid, 8*ns,  '*char'), 8,  ns)');
    nr_cell   = cellstr(reshape(fread(fid, 8*ns,  '*char'), 8,  ns)');
    samples_per_rec = cellfun(@(x) str2double(x), nr_cell);
    fs_all          = cellfun(@(x) str2double(x), fs_cell);

    % Pick first EEG channel
    chan = find(contains(labels, 'EEG', 'IgnoreCase', true), 1);
    if isempty(chan), chan = 1; end
    fs = fs_all(chan);

    % Read data records
    offset = 256 * (1 + ns);
    fseek(fid, offset, 'bof');
    total_samp = sum(samples_per_rec);
    raw = fread(fid, total_samp, 'int16');
    fclose(fid);

    % Extract channel
    idx = sum(samples_per_rec(1:chan-1)) + 1;
    sig = [];
    rec_size = sum(samples_per_rec);
    n_rec = floor(numel(raw)/rec_size);
    for r = 1:n_rec
        base = (r-1)*rec_size + idx;
        sig  = [sig; raw(base : base+samples_per_rec(chan)-1)]; %#ok<AGROW>
    end
    signal = double(sig);
end