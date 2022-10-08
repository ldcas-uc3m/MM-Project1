close all
clear
clc

%% Read Database
% Add local library and data to the path
addpath(genpath('lib'));
addpath(genpath('data'));
% Read raw data
disp('Reading database from local files...')
load('data/data.mat','X','y')
% Divide dataset into training and test set
[Xtrain,Ytrain,Xtest,Ytest] = divide_train_test(X,y);
disp('Local data read!')
fprintf('\n-Comedy films: %i',sum(y==0))
fprintf('\n-Drama  films: %i\n\n',sum(y==1))

%% Feature Extraction Stage
disp('Feature Extraction Stage in progress...')
% Creating empy array of features
features = zeros(length(Xtrain),5);
% features = [colour_entropy, brightness, edge_quantity, num_words, mean_word_length]

%% Visual Feature Extraction
disp('Extracting visual features...')
for i = 1:length(Xtrain)
    
    % Select current image
    I = Xtrain{i, 1};
    % imshow(I) to show the image
    %%% Feature 1: Dominant Colours
    % Convert I to HSV image
    HSV = rgb2hsv(I);
    % Select Hue component
    H = HSV(:,:,1);
    % Obtain the variability in colour (entropy)
    colour_entropy = entropy(H);
    % Save feature
    features(i,1) = colour_entropy;
    
    %%% Feature 2: Brightness
    % Extract the Value channel from HSV image
    V = HSV(:,:,3);
    % Obtain the mean value of the Value channel
    brightness = mean(V, 'all');
    % Save feature
    features(i,2) = brightness;
    
    %%% Feature 3: Edges
    % Convert I to gray-scale image
    Ig = rgb2gray(I);
    % Obtain edge image using the Sobel filter
    BW = edge(Ig, "Sobel");  % this returns an image with a 1 where there is an edge and a 0 otherwise
    % Get the amount of edges in the BW image
    im_size = size(BW);  % size of image, for normalization
    im_height = im_size(1);
    im_width = im_size(2);
    edge_quantity = sum(BW(:)) / (im_height * im_width);  % count the number of 1s, and normalize to size
    % Save feature
    features(i,3) = edge_quantity;    
  
end

%% Textual Feature Extraction
disp('Extracting textual features...')
for i = 1:length(Xtrain)
    
    % Select current text
    T = Xtrain(i, 2);
    % Tokenize document (separate into words)
    words = obtain_word_array(T);
    
    %%% Feature 4: Number of words
    % Obtain the number of words (tokens)
    num_words = length(words);
    % Save feature
    features(i,4) = num_words;  
    
    %%% Feature 5: Length of words
    % Obtain the length of each word in the description
    word_lengths = zeros(num_words, 1);
    for j = 1:num_words
        word_lengths(j) = strlength(words(j));

    end

    % Obtain the mean length of the words in the description
    mean_word_length = mean(word_lengths);
    % Save feature
    features(i,5) = mean_word_length; 

end
disp('Feature Extraction complete!')

%% Normalization Stage
disp('Normalization Stage in progress...')
% Obtain the mean of each feature
size_features = size(features);
num_features = size_features(2);

feat_mean = zeros(num_features, 1);

for i = 1:num_features
    feat_mean(i) = mean(features(:, i));
end

% Obtain the standard deviation of each feature
feat_std = zeros(num_features, 1);

for i = 1:num_features
    feat_std(i) = std(features(:, i));
end

% Normalize the extracted features
features_n = zeros(length(features), num_features);

for i = 1:length(features)
    for j = 1:num_features
        features_n(i, j) = (features(i, j) - feat_mean(j))/feat_std(j);
    end
end

% Check if normalization was correctly implemented (VERY IMPORTANT)
% If normalization was correctly implemented, running the line below should
% print the message saying so.
check_normalization(features_n);

%% Feature Visualization
% Select pair of features to visualize:
%   -1: Colour
%   -2: Brightness
%   -3: Edges
%   -4: Word number
%   -5: Word length
feat_a = .. ;
feat_b = .. ;
% Plot feature values in scatter diagram
figure()
visualize_features(features_n, Ytrain, feat_a, feat_b)

%% Training Stage
disp('Training Stage in progress...')
% Train model with all features available
model = fit_gaussian(features_n,Ytrain);
% Train model with just visual  features
visual_model = fit_gaussian(features_n(:,[1 2 3]),Ytrain);
% Train model with just textual features
textual_model = fit_gaussian(features_n(:,[4 5]),Ytrain);
disp('Training completed!')

%% Test Stage
disp('Testing Stage in progress...')
% IMPORTANT!!!
% Test images need to undergo the exact same process as training images
% Note that you can extract both types of features within the same loop
features_test = zeros(length(Xtest),5);

%% Test sample processing
for i = 1:length(Xtest)
 
    
    features_test(i,1) = .. ;
    features_test(i,2) = .. ;
    features_test(i,3) = .. ;
    features_test(i,4) = .. ;  
    features_test(i,5) = .. ; 
    
end

%% Test sample normalization
%%% Perform Normalization
% Note that you do not need to recompute the mean and standard deviation
% again. You need to use the values from training
features_test_n = .. ;

%% Test the models against the new extracted features
% Test visual  model
[labels_pred_v, scores_pred_v] = predict_gaussian(visual_model, ...
                                                  features_test_n(:,[1 2 3]));
% Test textual model
[labels_pred_t, scores_pred_t] = predict_gaussian(textual_model, ...
                                                  features_test_n(:,[4 5]));
% Test global  model
[labels_pred, scores_pred]     = predict_gaussian(model, ...
                                                  features_test_n);

%% Performance Assessment Stage
disp('Performance Assessment Stage in progress...')
labels_true = Ytest';
% Measure the performance of the developed system (Detection & False Alarm)
P_D  = .. ;
P_FA = .. ;

% Measure the performance of the developed system (AUC)
% (NO NEED TO CODE ANYTHING HERE)
[X1,Y1,T1,AUC1] = perfcurve(Ytest',scores_pred_v,1);
[X2,Y2,T2,AUC2] = perfcurve(Ytest',scores_pred_t,1);
[X3,Y3,T3,AUC3] = perfcurve(Ytest',scores_pred,1);
figure(2),area(X3,Y3,'FaceColor','Green','FaceAlpha',0.5)
hold on
figure(2),area(X3,X3,'FaceColor','White','FaceAlpha',0.7)
figure(2), plot(X3,Y3,'k','LineWidth',5)
figure(2), plot(X3,X3,'k--','LineWidth',5)
figure(2),area(X1,Y1,'FaceColor','Blue','FaceAlpha',0.5)
figure(2),area(X1,X1,'FaceColor','White','FaceAlpha',0.7)
figure(2), plot(X1,Y1,'k','LineWidth',5)
figure(2), plot(X1,X1,'k--','LineWidth',5)
figure(2),area(X2,Y2,'FaceColor','Red','FaceAlpha',0.5)
figure(2),area(X2,X2,'FaceColor','White','FaceAlpha',0.7)
figure(2), plot(X2,Y2,'k','LineWidth',5)
figure(2), plot(X2,X2,'k--','LineWidth',5)
title(['AUC (I) = ' num2str(AUC1) ' - AUC (T) = ' num2str(AUC2) ' - AUC (I+T) = ' num2str(AUC3)])
disp('Performance Assessed!')

save('data/features.mat','features','features_test','features_n','features_test_n');