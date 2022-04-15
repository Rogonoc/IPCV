function H = estimateTransformFeatures(imageA, imageB, ptThreshold, ptROI)
% H = estimateTransformFeatures(imageA, imageB, ptThreshold, ptROI)
%  Estimates the transformation matrix from image A to image B by first
%  detecting the SURF-features in both images, for which features are 
%  are extracted afterwards. Based on the common features the transformation
%  matrix is estimated from image A to image B.
%
%  imageA: first image acting as our base
%
%  imageB: second image to which we want to transform our base to
%
%  ptThreshold: threshold value for strength of found features
%       Integer value [0, ->]
%
%  ptROI: specify the Region of Interest over where features should be extracted
%       Rectangle of [cornerTopLeftX cornerTopLeftY width length]   
%
% BASED ON: cvexEstStabilizationTform() from MATLAB ©

% Set default parameters
if nargin < 4 || isempty(ptROI)
    ptROI = [1 1 size(imageA, 2) size(imageA, 1)];
end

%% Generate prospective points in image A and imageB
pointsA = detectSURFFeatures(imageA, 'MetricThreshold', ptThreshold, 'ROI', ptROI);
pointsB = detectSURFFeatures(imageB, 'MetricThreshold', ptThreshold, 'ROI', ptROI);

%% Select point correspondences
% Extract features for the corners
[featuresA, pointsA] = extractFeatures(imageA, pointsA);
[featuresB, pointsB] = extractFeatures(imageB, pointsB);

% Match features which were computed from the current and the previous
% images
indexPairs = matchFeatures(featuresA, featuresB);
pointsA = pointsA(indexPairs(:, 1), :);
pointsB = pointsB(indexPairs(:, 2), :);

%% Use MSAC algorithm to compute the projective transformation
tform = estimateGeometricTransform2D(pointsB, pointsA, 'projective');
H = tform.T;