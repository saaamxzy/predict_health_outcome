load('../training_data.mat')


[X_block_1, X_block_2, X] = feature_select(train_inputs);

m2 = mean(X_block_2);
X2_norm = bsxfun(@minus, X_block_2, m2);
sigma2 = std(X2_norm);
X2_norm = bsxfun(@rdivide, X2_norm, sigma2);

pcas = 500;

[coeff,score,latent] = pca(X2_norm);

errors = [];

pred_labels = [];

Lambda = logspace(-5, -1, 15);

Y = train_labels;

lr = logspace(-6, -2, 15);
for p = 1:1
    error_list = zeros(1, 15);
    train_error_list = zeros(1, 15);
    num_components = pcas;
    Xhat = score(:,1:num_components);
    Xhat = [X_block_1 Xhat];
    %for l = 1:size(Lambda,2)
    for l = 1:size(lr, 2)
        errors = [];
        train_errors = [];

        predictions = [];
        train_predictions = [];
        for j = 1:size(Y,2)
            Y_cur = Y(:,j);

            %B = fitrlinear(X_train_xval, Y_cur, 'Lambda', Lambda(l), 'Regularization', 'ridge');
            B = fitrlinear(Xhat, Y_cur,  'Regularization', 'ridge','Lambda', 0.007, 'LearnRate', lr(l));
            train_pred = predict(B, Xhat);
            train_predictions = [train_predictions, train_pred];

%             pred = predict(B, X_valid);
%             predictions = [predictions, pred];

            %pred_labels = [pred_labels, predict(B, X_valid)];

        end
        % errors = [errors, error_metric(predictions, Y_valid)];
        train_errors = [train_errors, error_metric(train_predictions, Y)];
        % error_list(l) = mean(errors);
        train_error_list(l) = mean(train_errors);
    end
%     plot(lr, error_list);
%     hold on;
    plot(lr, train_error_list);
    legend('train error');
    
    
end
