function [X_1, X_2, categoricals, X_out] = feature_select(X)
%FEATURE_SELECT Summary of this function goes here
%   Detailed explanation goes here
%X_1 = X(:,1:21);
X_1 = X(:,[1:6,12:20]);
X_2 = X(:,22:end);
categoricals = X(:,[7:11,21]);
X_out = X;
end

