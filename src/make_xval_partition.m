function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE
p = n / n_folds;
p = int16(p);
part = zeros(1, n);
count = 0;
for i = 1:n_folds
    if p*n_folds > n
        p = p - 1;
    end
    head = (i-1)*p;
    tail = i*p;
    part(head+1:tail) = i;
    count = count + p;
end
part(count+1:end) = randperm(n_folds, n - count);

idx = randperm(length(part));
part = part(idx);

end

