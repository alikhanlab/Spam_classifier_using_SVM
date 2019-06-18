function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = [1 2 1]; x2 = [0 4 -1];

c_choice = [0.01 0.03 0.1 0.3 1 3 10 30];

si_choice = [0.01 0.03 0.1 0.3 1 3 10 30];

[c_set, si_set] = ndgrid(c_choice, si_choice);

comb = horzcat(c_set(:), si_set(:));

results = [zeros(64, 1)];


% Models
for i = 1:64
    model = svmTrain(X, y, comb(i, 1), @(x1, x2) gaussianKernel(x1, x2, comb(i, 2)));
    predictions = svmPredict(model, Xval);
    results(i)= mean(double(predictions ~= yval));
end 

%disp('The results are: ')
%disp(results)

[val, pos] = min(results);

C = comb(pos, 1);
sigma = comb(pos, 2);

%disp('Index, C and sigma vlaues: ')
%disp(pos)
%disp(C)
%disp(sigma)

% =========================================================================

end
