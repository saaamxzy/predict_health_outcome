function pred_labels=k_means_predict_labels(train_inputs,train_labels,test_inputs)

pcas = 125;

% Train
[X_train_1, X_train_2, X_train_categoricals, X_train] = feature_select(train_inputs);
        
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

% Test

[X_test_1, X_test_2, X_test_categoricals, X_test] = feature_select(test_inputs);

X1_test_norm = bsxfun(@minus, X_test_1, m1);
X1_test_norm = bsxfun(@rdivide, X1_test_norm, sigma1);

X2_test_norm = bsxfun(@minus, X_test_2, m2);
X2_test_norm = bsxfun(@rdivide, X2_test_norm, sigma2);

Xhat_test = X2_test_norm * coeff(:,1:pcas);
o = ones(size(Xhat_test,1), 1);
Xhat_test = [X1_test_norm X_test_categoricals Xhat_test o];

predictions = [];
train_predictions = [];


k = 100;
[idx, C] = kmeans(Xhat, k);
            
valid_idx = [];
for index = 1:size(Xhat_test,1)
    x = Xhat_test(index,:);
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

Xhat_new = zeros(size(Xhat,1), k);
Xhat_test_new = zeros(size(Xhat_test,1), k);



for j = 1:size(Xhat_new, 1)
    Xhat_new(j,idx(j)) = 1;
end

for j = 1:size(Xhat_test_new, 1)
    Xhat_test_new(j,valid_idx(j)) = 1;
end

o = ones(size(Xhat_new,1), 1);
Xhat = [Xhat_new o];

o = ones(size(Xhat_test_new,1), 1);
Xhat_test = [Xhat_test_new o];
for j = 1:size(train_labels,2)      % 9 y
    Y_cur = train_labels(:,j);
    mean_Y = [];

    W = (Xhat' * Xhat + 0.07 * eye(size(Xhat,2))) \ (Xhat' * Y_cur);

    train_pred = Xhat*W;
    pred = Xhat_test*W;

    train_predictions = [train_predictions, train_pred];
    predictions = [predictions, pred];


end

pred_labels = predictions;
end

