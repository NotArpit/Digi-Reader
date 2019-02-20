function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% NOTE: n = 400
for i = 1:m
	% x => (n+1 x 1) (NOTE: n+1x1 not nx1 since we add the bias unit!):
	x = [1 X(i, :)]'; 
	% a2 => (25 x 401) x (n+1 x 1) => (26 x 1) (NOTE: Not 25x1 since we add the bias unit!):
	a2 = [1; sigmoid(Theta1 * x)];
	% a3 => (10 x 26) x (26 x 1) => (10 x 1) (NOTE: This is our hypothesis function so no
	% need to add the bias unit):
	a3 = [sigmoid(Theta2 * a2)];
	discrete_output = find(a3 == max(a3));
	p(i) = discrete_output;
end







% =========================================================================


end
