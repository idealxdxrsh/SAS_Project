function export_results(rec_name, pred_labels, true_labels, conf_matrix, ...
                        accuracy, kappa, spindles, kcomplexes, ...
                        eeg_clean, fs, features, feature_names, out_dir, fig_dir)
%EXPORT_RESULTS  Generate all output figures and CSV files
%
%  Figures produced:
%    1. Hypnogram (true vs predicted)
%    2. Confusion matrix heatmap
%    3. PSD comparison across sleep stages
%    4. Spectrogram (STFT) with hypnogram overlay
%    5. Spindle event gallery (up to 9 examples)
%    6. K-complex event gallery (up to 9 examples)
%    7. Feature violin plots by stage
%
%  CSVs produced:
%    - <rec>_epoch_labels.csv     (epoch-by-epoch true vs predicted)
%    - <rec>_spindles.csv         (spindle event table)
%    - <rec>_kcomplexes.csv       (K-complex event table)
%    - <rec>_features.csv         (full feature matrix)
% =========================================================================

    stage_order  = {'W','N1','N2','N3','R'};
    stage_colors = [0.85 0.33 0.10;   % W  - orange-red
                    0.00 0.45 0.74;   % N1 - blue
                    0.47 0.67 0.19;   % N2 - green
                    0.30 0.10 0.60;   % N3 - purple
                    0.93 0.69 0.13];  % R  - amber

    EPOCH_SEC  = 30;
    n_epochs   = numel(pred_labels);
    t_epochs   = (0:n_epochs-1) * EPOCH_SEC / 3600;   % hours

    y_true = label2int(true_labels,  stage_order);
    y_pred = label2int(pred_labels,  stage_order);

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 1: Hypnogram
    %% ════════════════════════════════════════════════════════════════════
    fig1 = figure('Visible','off','Position',[100 100 1100 320]);
    subplot(2,1,1)
        stairs(t_epochs, 6 - y_true, 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2);
        yticks(1:5); yticklabels(fliplr(stage_order));
        ylabel('True Stage'); xlabel('Time (hours)');
        title(sprintf('Hypnogram — %s', strrep(rec_name,'_',' ')));
        xlim([0 t_epochs(end)]); grid on; box off;
    subplot(2,1,2)
        stairs(t_epochs, 6 - y_pred, 'Color', [0.13 0.47 0.71], 'LineWidth', 1.2);
        yticks(1:5); yticklabels(fliplr(stage_order));
        ylabel('Predicted'); xlabel('Time (hours)');
        xlim([0 t_epochs(end)]); grid on; box off;
    tight_layout_fig(fig1);
    saveas(fig1, fullfile(fig_dir, [rec_name '_fig1_hypnogram.png']));
    close(fig1);

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 2: Confusion matrix
    %% ════════════════════════════════════════════════════════════════════
    fig2 = figure('Visible','off','Position',[100 100 480 420]);
    cm_norm = conf_matrix ./ (sum(conf_matrix,2) + eps);
    imagesc(cm_norm); colormap(gca, flipud(gray));
    colorbar; caxis([0 1]);
    xticks(1:5); xticklabels(stage_order);
    yticks(1:5); yticklabels(stage_order);
    xlabel('Predicted'); ylabel('True');
    title(sprintf('Confusion Matrix (Acc=%.1f%%, \\kappa=%.2f)', accuracy*100, kappa));
    for r=1:5
        for c=1:5
            text(c, r, sprintf('%.2f', cm_norm(r,c)), ...
                'HorizontalAlignment','center', 'FontSize', 9, ...
                'Color', cm_norm(r,c)>0.5*[1 1 1]);
        end
    end
    saveas(fig2, fullfile(fig_dir, [rec_name '_fig2_confusion.png']));
    close(fig2);

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 3: PSD per stage
    %% ════════════════════════════════════════════════════════════════════
    EPOCH_SAMP = EPOCH_SEC * fs;
    n_fft  = 2^nextpow2(2*fs*2);
    win_w  = hann(2*fs);
    fig3   = figure('Visible','off','Position',[100 100 650 420]);
    hold on;
    lgd = {};
    for s = 1:5
        idx_ep = find(y_true == s, 1);
        if isempty(idx_ep), continue; end
        seg = eeg_clean((idx_ep-1)*EPOCH_SAMP+1 : min(idx_ep*EPOCH_SAMP, numel(eeg_clean)));
        [pxx, f] = pwelch(seg, win_w, round(fs), n_fft, fs);
        f_mask = f <= 35;
        plot(f(f_mask), 10*log10(pxx(f_mask)+eps), ...
             'Color', stage_colors(s,:), 'LineWidth', 1.5);
        lgd{end+1} = stage_order{s}; %#ok<AGROW>
    end
    xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    title('Power Spectral Density by Sleep Stage');
    legend(lgd, 'Location','northeast'); grid on; box off;
    saveas(fig3, fullfile(fig_dir, [rec_name '_fig3_psd.png']));
    close(fig3);

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 4: Spectrogram with hypnogram strip
    %% ════════════════════════════════════════════════════════════════════
    fig4 = figure('Visible','off','Position',[100 100 1100 500]);
    % Use first 2 hours max for speed
    max_samp = min(numel(eeg_clean), 2*3600*fs);
    seg_sg   = eeg_clean(1:max_samp);
    win_sg   = hann(2*fs);
    hop_sg   = round(0.5*fs);
    [S, F, T_sg] = spectrogram(seg_sg, win_sg, numel(win_sg)-hop_sg, [], fs);
    f_mask   = F <= 35;

    subplot(5,1,1:4)
        imagesc(T_sg/3600, F(f_mask), 10*log10(abs(S(f_mask,:)).^2 + eps));
        axis xy; colormap(gca, jet);
        clim_val = prctile(10*log10(abs(S(f_mask,:)).^2 + eps), [5 98], 'all');
        clim(clim_val);
        ylabel('Freq (Hz)'); title('EEG Spectrogram');
        colorbar;

    subplot(5,1,5)
        n_ep_shown = min(n_epochs, round(max_samp/(EPOCH_SAMP)));
        stairs((0:n_ep_shown-1)*EPOCH_SEC/3600, ...
               6 - y_pred(1:n_ep_shown), 'k', 'LineWidth', 1.2);
        yticks(1:5); yticklabels(fliplr(stage_order));
        ylabel('Stage'); xlabel('Time (hours)');
        xlim([0 max_samp/(fs*3600)]);

    saveas(fig4, fullfile(fig_dir, [rec_name '_fig4_spectrogram.png']));
    close(fig4);

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 5: Spindle gallery
    %% ════════════════════════════════════════════════════════════════════
    if ~isempty(spindles)
        fig5 = figure('Visible','off','Position',[100 100 900 700]);
        n_show = min(9, numel(spindles));
        suptitle_str = sprintf('Detected Sleep Spindles (n=%d shown)', n_show);
        for k = 1:n_show
            subplot(3,3,k);
            i0 = max(1, round(spindles(k).onset_s  * fs) - round(0.3*fs));
            i1 = min(numel(eeg_clean), round(spindles(k).offset_s * fs) + round(0.3*fs));
            t_seg = (i0:i1)/fs - spindles(k).onset_s;
            plot(t_seg, eeg_clean(i0:i1), 'b', 'LineWidth', 0.8);
            xline(0, 'r--'); xline(spindles(k).duration_s, 'r--');
            title(sprintf('%.2f s | %.1f Hz', spindles(k).duration_s, spindles(k).freq_hz));
            xlabel('Time (s)'); ylabel('Amp');
            grid on; box off;
        end
        sgtitle(suptitle_str);
        saveas(fig5, fullfile(fig_dir, [rec_name '_fig5_spindles.png']));
        close(fig5);
    end

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 6: K-complex gallery
    %% ════════════════════════════════════════════════════════════════════
    if ~isempty(kcomplexes)
        fig6 = figure('Visible','off','Position',[100 100 900 700]);
        n_show = min(9, numel(kcomplexes));
        for k = 1:n_show
            subplot(3,3,k);
            i0 = max(1, round(kcomplexes(k).onset_s  * fs) - round(0.5*fs));
            i1 = min(numel(eeg_clean), round(kcomplexes(k).offset_s * fs) + round(0.5*fs));
            t_seg = (i0:i1)/fs - kcomplexes(k).onset_s;
            plot(t_seg, eeg_clean(i0:i1), 'Color',[0.5 0 0.5], 'LineWidth', 0.8);
            xline(0,'r--'); xline(kcomplexes(k).duration_s,'r--');
            title(sprintf('Dur=%.2f s', kcomplexes(k).duration_s));
            xlabel('Time (s)'); ylabel('Amp');
            grid on; box off;
        end
        sgtitle(sprintf('Detected K-Complexes (n=%d shown)', n_show));
        saveas(fig6, fullfile(fig_dir, [rec_name '_fig6_kcomplexes.png']));
        close(fig6);
    end

    %% ════════════════════════════════════════════════════════════════════
    %  FIG 7: Feature distributions (box plots for top 5 features)
    %% ════════════════════════════════════════════════════════════════════
    fig7 = figure('Visible','off','Position',[100 100 1000 400]);
    top_feats = [1 2 6 7 11];   % delta_abs, theta_abs, delta_rel, theta_rel, SEF
    for fi = 1:numel(top_feats)
        subplot(1, numel(top_feats), fi);
        feat_data = cell(5,1);
        for s = 1:5
            feat_data{s} = features(y_true==s, top_feats(fi));
        end
        boxplot(cell2mat(feat_data), repelem((1:5)', cellfun(@numel,feat_data)));
        xticklabels(stage_order);
        title(strrep(feature_names{top_feats(fi)},'_',' '), 'FontSize',8);
        grid on; box off;
    end
    sgtitle('Feature Distributions by Sleep Stage');
    saveas(fig7, fullfile(fig_dir, [rec_name '_fig7_features.png']));
    close(fig7);

    %% ════════════════════════════════════════════════════════════════════
    %  CSV EXPORTS
    %% ════════════════════════════════════════════════════════════════════

    % 1. Epoch labels
    ep_table = table((1:n_epochs)', true_labels(:), pred_labels(:), ...
                     strcmp(true_labels(:), pred_labels(:)), ...
                     'VariableNames', {'Epoch','TrueStage','PredStage','Correct'});
    writetable(ep_table, fullfile(out_dir, [rec_name '_epoch_labels.csv']));

    % 2. Spindles
    if ~isempty(spindles)
        sp_table = struct2table(spindles);
        writetable(sp_table, fullfile(out_dir, [rec_name '_spindles.csv']));
    end

    % 3. K-complexes
    if ~isempty(kcomplexes)
        kc_table = struct2table(kcomplexes);
        writetable(kc_table, fullfile(out_dir, [rec_name '_kcomplexes.csv']));
    end

    % 4. Feature matrix
    feat_table = array2table(features, 'VariableNames', ...
                             matlab.lang.makeValidName(feature_names));
    feat_table.Stage = true_labels(:);
    writetable(feat_table, fullfile(out_dir, [rec_name '_features.csv']));

    fprintf('    Figures and CSVs saved to: %s\n', out_dir);
end


%% ── Helpers ──────────────────────────────────────────────────────────────
function y = label2int(labels, order)
    y = zeros(numel(labels), 1);
    for i = 1:numel(labels)
        idx = find(strcmp(order, labels{i}), 1);
        if isempty(idx), idx = 2; end
        y(i) = idx;
    end
end

function tight_layout_fig(fig)
    % Tighten subplot spacing
    set(findall(fig, 'type', 'axes'), 'FontSize', 9);
end
