load('./training_data.mat')

% Put your train and test data here:
train_inputs = train_inputs; % TODO: modify this to your data
train_labels = train_labels; % TODO: modify this to your data
test_inputs = train_inputs;  % TODO: modify this to your data

% ****** Linear Regression with Ridge regularization ****** %
pred_labels = lr_predict_labels(train_inputs,train_labels,test_inputs);
lr_error = error_metric(train_labels, pred_labels)

% ****** K means regression ****** %
pred_labels = k_means_predict_labels(train_inputs,train_labels,test_inputs);
km_error = error_metric(train_labels, pred_labels)


% ****** K nearest neighbors ****** %
pred_labels = knn_predict_labels(train_inputs,train_labels,test_inputs);

knn_error = error_metric(train_labels, pred_labels)

% ****** Ensemble ****** %
pred_labels = ensemble_predict_labels(train_inputs,train_labels,test_inputs);

ens_error = error_metric(train_labels, pred_labels)