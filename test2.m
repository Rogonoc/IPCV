%% Reset workspace
clear;
clc;

%% Stabilize
% Load video to variable
hVideoSrc = VideoReader('MAH01462.MP4');

% User input
roi = [0.500000000000000,0.500000000000000,1440,5.257500000000003e+02]; % Horizontal looking island
roi_buoy = [6.432499999999999e+02, 5.00000000000001e+02, 35.500000000000000, 35.500000000000000]; % Initial ROI surrounding buoy; based on first frame
mThreshold = 1300;

% Reset the video source to the beginning of the file.
read(hVideoSrc, 1);

% Create video viewer
hVPlayer = vision.VideoPlayer; 

% Create tracker object
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'BlockSize', [5 5]);

% Initialize frames
imgB = rgb2gray(im2single(readFrame(hVideoSrc)));
imgBp = imgB;

% Initialize other variables
trackerAlive = 0;
KLT_points = [];

% Find features of interest in first frame
points = detectBRISKFeatures(imgBp, 'MinQuality', 0.6, 'MinContrast', 0.3, 'ROI', roi_buoy);
%points = detectMinEigenFeatures(imgBp, 'MinQuality', 0.9, 'ROI', roi_buoy);
%pointImage = insertMarker(imgBp, points.Location, '+', 'Color', 'white');
%pointImage = insertShape(pointImage, 'Rectangle', roi_buoy, 'Color','red');
%figure(2); imshow(pointImage);

% Initialize the tracker
%initialize(tracker, points.Location, imgBp);

% Starting variable
ii = 2;

% Initial transformation matrix
Hcumulative = eye(3);

% ALGORITHM
% 1. Initialize ROI
% 2. Wait till FeatureFinder detects point in ROI. If yes, initialize KLT-object
% 3. Using ROI around returned point of KLT-object, check if the FeatureFinder matches with this point
% 4. If no point is found in FeatureFinder, the buoy is probably missing. Disable KLT-object
% 5. Freeze ROI. Go back to step 2

% Start tracking features in video
while hasFrame(hVideoSrc) && ii < 200
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
    %step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));
    frame = imgBp;

    % Check if buoy is detected in ROI using FeatureFinder
    if (trackerAlive == 0)
        % Detect features using BRISK-algorithm 
        points = detectBRISKFeatures(imgBp, 'MinQuality', 0.6, 'MinContrast', 0.28, 'ROI', roi_buoy);

        % Check if points were returned by FeatureFinder in ROI
        if ~isempty(points) 
            % Initialize the KLT-tracker on the found point
            initialize(tracker, points.Location, frame);
            
            % Return points of the KLT-tracker
            [KLT_points, validity] = tracker(frame);
    
            % Insert marker in frame to indicate (potential) buoy
            frame = insertMarker(frame, KLT_points(validity, :), '+', 'Color', 'white');
    
            % Convert frame to correct dimensions
            frame = rgb2gray(im2single(frame));
    
            % Indicate that tracker is alive
            trackerAlive = 1;
        end
    elseif (trackerAlive == 1)
        % DON'T INITIALIZE TRACKER AGAIN
        % LET IT DO ITS THING

        % Return points of the KLT-tracker
        [KLT_points, validity] = tracker(frame);

        % Check ROI based on current returned point of KLT-tracker
        rectSize = 10;
        points = detectBRISKFeatures(imgBp, 'MinQuality', 0.2, 'MinContrast', 0.20, 'ROI', [KLT_points(1) - rectSize, KLT_points(2) - rectSize, 2*rectSize, 2*rectSize]);

        % Are there no points returned? --> tracker should not be alive anymore
        if (isempty(KLT_points))
            release(tracker); % Release the tracker
            trackerAlive = 0; % Disable tracker
        else
            % Insert marker in frame to indicate (potential) buoy
            frame = insertMarker(frame, KLT_points(validity, :), '+', 'Color', 'white');

            % Convert frame to correct dimensions
            frame = rgb2gray(im2single(frame));
        end
    end
    
    % Play current frame in video player
    step(hVPlayer, frame);

    % Increment frame counter
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
roi_buoy = [6.432499999999999e+02, 5.00000000000001e+02, 35.500000000000000, 35.500000000000000];
objectImage = insertShape(imgBp, 'Rectangle', roi_buoy, 'Color','red');
figure(1); imshow(objectImage)

%%

% Detect points in ROI
points = detectBRISKFeatures(imgBp, 'MinQuality', 0.6, 'MinContrast', 0.275, 'ROI', roi_buoy);
%points = detectMinEigenFeatures(imgBp, 'MinQuality', 0.99, 'ROI', roi_buoy);
%points = detectSURFFeatures(im2gray(imgBp), 'MetricThreshold', 0, 'ROI', roi_buoy);
pointImage = insertMarker(imgBp, points.Location, '+', 'Color', 'white');
figure(2); imshow(pointImage);