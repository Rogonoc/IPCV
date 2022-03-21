function H = estimateTransformFeatures(imageA, imageB, ptThreshold, ptROI)

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

%% Use MSAC algorithm to compute the affine transformation
tform = estimateGeometricTransform2D(pointsB, pointsA, 'projective');
H = tform.T;