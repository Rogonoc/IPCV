%% Reset workspace
clear;
clc;

%% Stabilize
% Load video to variable
hVideoSrc = VideoReader('MAH01462.MP4');

% User input
roi = [0.500000000000000,0.500000000000000,1440,5.257500000000003e+02]; % Horizontal looking island
mThreshold = 1300;

% Reset the video source to the beginning of the file.
read(hVideoSrc, 1);

% Create video viewer
hVPlayer = vision.VideoPlayer; 

% Initialize moving and corrected mean
imgB = rgb2gray(im2single(readFrame(hVideoSrc)));
imgBp = imgB;
correctedMean = imgBp;

% Starting variable
ii = 2;

% Initial transformation matrix
Hcumulative = eye(3);

while hasFrame(hVideoSrc) && ii < 3
    % Read new frame
    imgA = imgB; % z^-1
    frameA = readFrame(hVideoSrc);
    imgAp = imgBp; % z^-1
    imgB = rgb2gray(im2single(readFrame(hVideoSrc)));

    % Estimate transform from current frame A to next frame B
    H = estimateTransformFeatures(imgA, imgB, 1500, roi);

    % Compute cumulative transformation matrix
    Hcumulative = H * Hcumulative;

    % Warp image
    imgBp = imwarp(imgB, projective2d(Hcumulative), 'OutputView', imref2d(size(imgB)));
    
    % Display as color composite with last corrected frame
    %step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));
    frame = readFrame(hVideoSrc);
    step(hVPlayer,frame);

    % Compute corrected mean
    correctedMean = correctedMean + imgBp;

    % Incrmeent frame counter
    ii = ii + 1;
end

% Release video viewer
release(hVPlayer);

%% Test

roi = [0.5, 402, 1042, 90]; % Horizontal looking island

imgA = rgb2gray(im2single(readFrame(hVideoSrc))); % Read first frame into imgA
pointsA = detectSURFFeatures(imgA, 'MetricThreshold', 2000, 'ROI', roi);
% Display corners found in images A and B.
figure; imshow(imgA); hold on;
plot(pointsA);
title('Corners in A');

%% Test 2

% show image
%imshow(imgA)

% Draw ROI over location of buoy 
%roi_buoy = [6.432499999999999e+02, 5.00000000000001e+02, 34.500000000000000, 25.500000000000000];
roi_buoy = [6.432499999999999e+02, 5.00000000000001e+02, 34.500000000000000, 25.500000000000000];
objectImage = insertShape(imgBp, 'Rectangle', roi_buoy, 'Color','red');
figure(1); imshow(objectImage)

%%

% Detect points in ROI
points = detectMinEigenFeatures(imgBp, 'MinQuality', 0.5, 'ROI', roi_buoy);
%points = detectSURFFeatures(im2gray(imgBp), 'MetricThreshold', 0, 'ROI', roi_buoy);
pointImage = insertMarker(imgBp, points.Location, '+', 'Color', 'white');
figure(2); imshow(pointImage);