load('./training_data.mat')

%pred_labels = [];                                                  % pred labels
Lambda = [0.1, 0.3, 0.5, 0.7, 0.9];               % penalties
lr = logspace(-5, 2, 15);                                           % learning rates
pcas = 125;                                                         % num of pcas
part = make_xval_partition(size(train_inputs, 1), 5);
N = max(part);
error_list = zeros(1, size(Lambda, 2));
train_error_list = zeros(1, size(Lambda,2));

Y = train_labels;

for l = 1:size(Lambda,2)
    errors = [];                                                    % final errors
    train_errors = [];
    
    for i = 1:N
    
        train_idx = part ~= i;
        valid_idx = part == i;

        X_train_xval = train_inputs(train_idx,:);
        Y_train_xval = train_labels(train_idx,:);
        
        [X_train_1, X_train_2, X_train_categoricals, X_train] = feature_select(X_train_xval);
        
        m1 = mean(X_train_1);
        X1_norm = bsxfun(@minus, X_train_1, m1);
        sigma1 = std(X1_norm);
        X1_norm = bsxfun(@rdivide, X1_norm, sigma1);

        m2 = mean(X_train_2);
        X2_norm = bsxfun(@minus, X_train_2, m2);
        sigma2 = std(X2_norm);
        X2_norm = bsxfun(@rdivide, X2_norm, sigma2);

        [coeff,score_train,latent] = pca(X2_norm);
        
        Xhat = score_train(:,1:pcas);
        o = ones(size(Xhat,1), 1);
        Xhat = [X1_norm X_train_categoricals Xhat o];
        
        % Validation
        X_validation = train_inputs(valid_idx,:);
        Y_valid = train_labels(valid_idx,:);
        
        [X_valid_1, X_valid_2, X_valid_categoricals, X_valid] = feature_select(X_validation);
        
        X1_valid_norm = bsxfun(@minus, X_valid_1, m1);
        X1_valid_norm = bsxfun(@rdivide, X1_valid_norm, sigma1);
        
        X2_valid_norm = bsxfun(@minus, X_valid_2, m2);
        X2_valid_norm = bsxfun(@rdivide, X2_valid_norm, sigma2);
        
        
        Xhat_valid = X2_valid_norm * coeff(:,1:pcas);
        o = ones(size(Xhat_valid,1), 1);
        Xhat_valid = [X1_valid_norm X_valid_categoricals Xhat_valid o];

        predictions = [];
        train_predictions = [];
        
        B1_w = 1;
        B2_w = 1.0 - B1_w;
        
        for j = 1:size(Y_train_xval,2)
            Y_cur = Y_train_xval(:,j);

%             B1 = fitrensemble(Xhat,Y_cur, 'Method','Bag','NumLearningCycles',100,'Learners', templateTree('MaxNumSplits',50));
%             B2 = fitrlinear(Xhat, Y_cur, 'Regularization', 'ridge');
            %B = fitrlinear(X_train_xval, Y_cur, 'Regularization', 'ridge', 'Lambda', 0.03, 'LearnRate', lr(l));
%             train_pred = ( B1_w * predict(B1, Xhat) + B2_w * predict(B2, Xhat) );
%             train_predictions = [train_predictions, train_pred];
% 
%             pred = ( B1_w * predict(B1, Xhat_valid) + B2_w * predict(B2, Xhat_valid) );
%             predictions = [predictions, pred];

            W = (Xhat' * Xhat + Lambda(l) * eye(size(Xhat,2))) \ (Xhat' * Y_cur);

            train_pred = Xhat*W;
            pred = Xhat_valid*W;
            
            %train_pred = predict(B1, Xhat);
            train_predictions = [train_predictions, train_pred];
            %pred = predict(B1, Xhat_valid);
            predictions = [predictions, pred];

            %pred_labels = [pred_labels, predict(B, X_valid)];
%                 X_train_xval = Xhat(train_idx,:);
%                 X_valid = Xhat(valid_idx,:);

        end
        errors = [errors, error_metric(predictions, Y_valid)];
        train_errors = [train_errors, error_metric(train_predictions, Y_train_xval)];
        
                
    end
    error_list(l) = mean(errors)
    train_error_list(l) = mean(train_errors)
    
end
plot(Lambda, train_error_list);
hold on;
plot(Lambda, error_list);
hold off;
legend('Training error', 'CV error');
xlabel('L2 Penalty Coefficient');
ylabel('errors');

