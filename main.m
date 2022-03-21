%% Reset workspace
clear;
clc;

%% Stabilize
% Load video to variable
hVideoSrc = VideoReader('MAH01462.MP4');

% User input
roi = [0.5, 402, 1042, 90]; % Horizontal looking island
mThreshold = 1500;

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

while hasFrame(hVideoSrc) && ii < hVideoSrc.NumFrames
    % Read new frame
    imgA = imgB; % z^-1
    imgAp = imgBp; % z^-1
    imgB = rgb2gray(im2single(readFrame(hVideoSrc)));

    % Estimate transform from current frame A to next frame B
    H = estimateTransformFeatures(imgA, imgB, 1500, roi);

    % Compute cumulative transformation matrix
    Hcumulative = H * Hcumulative;

    % Warp image
    imgBp = imwarp(imgB, projective2d(Hcumulative), 'OutputView', imref2d(size(imgB)));
    
    % Display as color composite with last corrected frame
    step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));

    % Compute corrected mean
    correctedMean = correctedMean + imgBp;

    % Incrmeent frame counter
    ii = ii + 1;
end

% Release video viewer
release(hVPlayer);

%% Test
% 
% roi = [0.5, 402, 1042, 90]; % Horizontal looking island
% 
% imgA = rgb2gray(im2single(readFrame(hVideoSrc))); % Read first frame into imgA
% pointsA = detectSURFFeatures(imgA, 'MetricThreshold', 1500, 'ROI', roi);
% % Display corners found in images A and B.
% figure; imshow(imgA); hold on;
% plot(pointsA);
% title('Corners in A');