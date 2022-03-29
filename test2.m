%% Reset workspace
clear;
clc;

%% Stabilize
% Load video to variable
hVideoSrc = VideoReader('MAH01462.MP4');

% User input
roi = [0.5,0.5,1440,5.2575e+02];                      % Horizontal looking island + clouds
roi_buoy_initial = [6.4325e+02, 5.0e+02, 35.5, 35.5]; % Initial ROI surrounding buoy; based on first frame
roi_buoy_tracker = 15;                                % Size of scanning ROI-region surrounding the KLT-tracker
mThreshold = 1300;                                    % Strictness of feature extraction for stabilization transforms

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
trackerAlive = 0;     % State of KLT-tracker (1)
trackerWasAlive = 0;  % State of KLT-tracker (2)
KLT_points = [];      % Points spit out by KLT-tracker
ii = 2;               % Loop variable
Hcumulative = eye(3); % Initial transformation matrix

% Find features of interest in first frame
points = detectBRISKFeatures(imgBp, 'MinQuality', 0.6, 'MinContrast', 0.3, 'ROI', roi_buoy_initial);

% ALGORITHM
% 1. Initialize ROI
% 2. Wait till FeatureFinder detects point in ROI. If yes, initialize KLT-object
% 3. Let KLT-tracker do its own thing
% 4. If no point has been returned by KLT-tracker, the buoy is probably missing. Disable KLT-object
% 5. Freeze ROI around "dead" point. Go back to step 2

% Start tracking features in video
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
    %step(hVPlayer, imfuse(imgAp, imgBp, 'ColorChannels', 'red-cyan'));
    frame = imgBp;

    % Check if buoy is detected in ROI using FeatureFinder
    if (trackerAlive == 0) % Tracker not active

        % Detect features using BRISK-algorithm 
        if (trackerWasAlive == 0)     % Use initial knowledge of ROI around buoy to start tracking it [STRICT]
            points = detectBRISKFeatures(frame, 'MinQuality', 0.6, 'MinContrast', 0.3, 'ROI', roi_buoy_initial);
        elseif (trackerWasAlive == 1) % Use previous knowledge of KLT-tracker point to cast new ROI around it [SEMI-STRICT]
            points = detectBRISKFeatures(frame, 'MinQuality', 0.2, 'MinContrast', 0.2, 'ROI', [KLT_points(1) - floor(roi_buoy_tracker/2), KLT_points(2) - floor(roi_buoy_tracker/2), 2*roi_buoy_tracker, 2*roi_buoy_tracker]);
        end

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
            trackerWasAlive = 1;
        end

    elseif (trackerAlive == 1) % Tracker active

        % DON'T INITIALIZE TRACKER AGAIN
        % LET IT DO ITS THING
        
        % Return points of the KLT-tracker
        [KLT_points, validity] = tracker(frame);

        % Check ROI based on current returned point of KLT-tracker [WEAK]
        points = detectBRISKFeatures(frame, 'MinQuality', 0.2, 'MinContrast', 0.1, 'ROI', [KLT_points(1) - (roi_buoy_tracker/2), KLT_points(2) - (roi_buoy_tracker/2), 2*roi_buoy_tracker, 2*roi_buoy_tracker]);

        % Are there no points returned? --> tracker should not be alive anymore
        if (isempty(KLT_points) || isempty(points))
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