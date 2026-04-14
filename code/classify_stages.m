function [pred_labels, conf_matrix, accuracy, kappa] = classify_stages(features, true_labels)
%CLASSIFY_STAGES  Train and evaluate a sleep stage classifier
%
%  Uses 5-fold stratified cross-validation.
%  Three classifiers are trained and the majority vote is taken:
%    1. K-Nearest Neighbours (k=7, cosine distance)
%    2. Decision Tree (max depth 12)
%    3. Linear SVM (one-vs-rest, L2 regularised)
%
%  Inputs:
%    features    - [num_epochs x num_features] feature matrix
%    true_labels - cell array of stage strings {'W','N1','N2','N3','R'}
%
%  Outputs:
%    pred_labels - cell array of predicted stage strings
%    conf_matrix - [5 x 5] confusion matrix (rows=true, cols=pred)
%    accuracy    - scalar 0–1
%    kappa       - Cohen's Kappa coefficient
% =========================================================================

    stage_order = {'W','N1','N2','N3','R'};
    n_stages    = numel(stage_order);

    % ── Encode labels to integers ─────────────────────────────────────────
    y_true = label2int(true_labels, stage_order);

    % ── Feature normalisation (z-score per feature) ───────────────────────
    [X_norm, mu, sg] = zscore(features);
    X_norm(isnan(X_norm)) = 0;
    X_norm(isinf(X_norm)) = 0;

    n_samples = size(X_norm, 1);
    y_pred    = zeros(n_samples, 1);

    %% ── 5-fold stratified cross-validation ──────────────────────────────
    K = 5;
    cv = cvpartition(y_true, 'KFold', K, 'Stratify', true);

    fprintf('    Running %d-fold stratified CV...\n', K);

    all_preds = zeros(n_samples, 3);   % 3 classifiers

    for fold = 1:K
        tr_idx = training(cv, fold);
        te_idx = test(cv, fold);

        X_tr = X_norm(tr_idx, :);
        y_tr = y_true(tr_idx);
        X_te = X_norm(te_idx, :);

        % ── 1. KNN ───────────────────────────────────────────────────────
        knn_mdl = fitcknn(X_tr, y_tr, 'NumNeighbors', 7, ...
                          'Distance', 'cosine', 'Standardize', false);
        all_preds(te_idx, 1) = predict(knn_mdl, X_te);

        % ── 2. Decision Tree ─────────────────────────────────────────────
        dt_mdl  = fitctree(X_tr, y_tr, 'MaxNumSplits', 2^12-1, ...
                           'MinLeafSize', 3);
        all_preds(te_idx, 2) = predict(dt_mdl, X_te);

        % ── 3. SVM (one-vs-rest via fitcecoc) ────────────────────────────
        t_svm   = templateSVM('KernelFunction','linear','BoxConstraint',1);
        svm_mdl = fitcecoc(X_tr, y_tr, 'Learners', t_svm);
        all_preds(te_idx, 3) = predict(svm_mdl, X_te);

        fprintf('      Fold %d complete.\n', fold);
    end

    %% ── Majority vote ────────────────────────────────────────────────────
    for i = 1:n_samples
        votes = all_preds(i, :);
        y_pred(i) = mode(votes);
    end

    %% ── Metrics ──────────────────────────────────────────────────────────
    conf_matrix = confusionmat(y_true, y_pred, 'Order', 1:n_stages);
    accuracy    = sum(diag(conf_matrix)) / sum(conf_matrix(:));
    kappa       = cohen_kappa(conf_matrix);

    % Per-class metrics
    fprintf('\n    %-6s  %8s  %8s  %8s\n','Stage','Precision','Recall','F1');
    fprintf('    %s\n', repmat('-',1,36));
    for s = 1:n_stages
        tp = conf_matrix(s,s);
        fp = sum(conf_matrix(:,s)) - tp;
        fn = sum(conf_matrix(s,:)) - tp;
        prec = tp / (tp + fp + eps);
        rec  = tp / (tp + fn + eps);
        f1   = 2*prec*rec / (prec + rec + eps);
        fprintf('    %-6s  %8.3f  %8.3f  %8.3f\n', stage_order{s}, prec, rec, f1);
    end
    fprintf('    %s\n', repmat('-',1,36));
    fprintf('    Overall Accuracy : %.1f%%\n', accuracy*100);
    fprintf('    Cohen''s Kappa   : %.3f\n',   kappa);

    %% ── Back to label strings ────────────────────────────────────────────
    pred_labels = int2label(y_pred, stage_order);
end


%% ── Helpers ──────────────────────────────────────────────────────────────
function y = label2int(labels, order)
    y = zeros(numel(labels), 1);
    for i = 1:numel(labels)
        idx = find(strcmp(order, labels{i}), 1);
        if isempty(idx), idx = 2; end   % default N1
        y(i) = idx;
    end
end

function labels = int2label(y, order)
    labels = cell(numel(y), 1);
    for i = 1:numel(y)
        idx = y(i);
        if idx < 1 || idx > numel(order), idx = 2; end
        labels{i} = order{idx};
    end
end

function k = cohen_kappa(C)
    n   = sum(C(:));
    p_o = sum(diag(C)) / n;
    p_e = sum(sum(C,2) .* sum(C,1)') / n^2;
    k   = (p_o - p_e) / (1 - p_e + eps);
end
