load('../training_data.mat')

%pred_labels = [];               % pred labels
Lambda = linspace(0.007, 0.5, 20);               % penalties
lr = logspace(-5, 2, 15);                   % learning rates
pcas = 100;                                 % num of pcas
part = make_xval_partition(size(train_inputs, 1), 5);
N = max(part);
error_list = zeros(1, 15);
train_error_list = zeros(1, 15);
options = statset('MaxIter',1000); % Increase number of EM iterations
K = [10, 20, 30, 50, 80, 100]; 

Y = train_labels;

%for l = 1:size(Lambda,1)
for k = 1:size(K,2)
    errors = [];                    % final errors
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
        
        [idx, C] = kmeans(Xhat, K(k));
            
        valid_idx = [];
        for index = 1:size(Xhat_valid,1)
            x = Xhat_valid(index,:);
            nearest = 0;
            nearest_dist = realmax;
            for c_index = 1:size(C,1)
                clus = idx(c_index);
                L2_dist = norm(x - C(c_index,:), 2);
                if L2_dist < nearest_dist
                    nearest = clus;
                    nearest_dist = L2_dist;
                end
            end
            valid_idx = [valid_idx, nearest];
        end

        Xhat_new = zeros(size(Xhat,1), K(k));
        Xhat_valid_new = zeros(size(Xhat_valid,1), K(k));

        
        
        for j = 1:size(Xhat_new, 1)
            Xhat_new(j,idx(j)) = 1;
        end
        
        for j = 1:size(Xhat_valid_new, 1)
            Xhat_valid_new(j,valid_idx(j)) = 1;
        end
        
        o = ones(size(Xhat_new,1), 1);
        Xhat = [Xhat_new o];
        
        o = ones(size(Xhat_valid_new,1), 1);
        Xhat_valid = [Xhat_valid_new o];
        for j = 1:size(Y_train_xval,2)      % 9 y
            Y_cur = Y_train_xval(:,j);
            mean_Y = [];

            W = (Xhat' * Xhat + 0.07 * eye(size(Xhat,2))) \ (Xhat' * Y_cur);

            train_pred = Xhat*W;
            pred = Xhat_valid*W;
            
            train_predictions = [train_predictions, train_pred];
            predictions = [predictions, pred];


        end
        errors = [errors, error_metric(predictions, Y_valid)];
        train_errors = [train_errors, error_metric(train_predictions, Y_train_xval)];
        
                
    end
    error_list(l) = mean(errors)
    train_error_list(l) = mean(train_errors)
    
end
