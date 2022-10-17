% test movie with id 18 (sum of student ids: 200858018)
% NOTE: run this AFTER Training Stage in script.m
[labels, scores] = predict_gaussian(model, features_n(18, :));
% returns 0 (comedy), with a score of -2.3605
Ytrain(18)  % the model was right
imshow(Xtrain{18, 1})  % the movie