function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


a = -y' * arrayfun(@(x) log(x), sigmoid(theta' * X')');
b = (ones(size(y)) - y)';
c = arrayfun(@(x) log(x),ones(size(sigmoid(theta' * X'))) - sigmoid(theta' * X'));
J = 1 / m * (a - b * c') + lambda / (2 * m) * (theta' .^2 * ones(size(theta)) - theta(1) ^ 2);
grad = ((sigmoid(theta' * X')' - y)' * X) ./ m + lambda / m * theta';
grad(1) = grad(1) -lambda / m * theta(1);



% =============================================================

end
