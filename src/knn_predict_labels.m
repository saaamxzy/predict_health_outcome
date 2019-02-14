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

% For ensemble, not used for now
B1_w = 0.5;
B2_w = 0.5;
% L2 penalty
L2 = 0.3;
k = 100;


for j = 1:size(train_labels,2)      % 9y
    Y_cur = train_labels(:,j);
    mean_Y = [];

    mdl = fitcknn(Xhat, Y_cur, 'NumNeighbors',5);

    train_pred = predict(mdl, Xhat);
    pred = predict(mdl, Xhat_test);

    train_predictions = [train_predictions, train_pred];
    predictions = [predictions, pred];


end

pred_labels = predictions;
end

