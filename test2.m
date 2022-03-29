%% Contact information
% Authors: 
% Year: 2022
% License: MIT
% Description: Tracking a buoy in a "Man-Over-Board" video using
%              the provided MATLAB KLT-algorithm. Image stabilization
%              has been used as preprocessing.

%% Reset workspace
clear; clc;

%% Stabilize image + tracking buoy

% Load video to variable
hVideoSrc = VideoReader('MAH01462.MP4');

% Reset the video source to the beginning of the file.
read(hVideoSrc, 1);

% User input
roi = [0.5,0.5,1440,5.2575e+02];                      % Horizontal looking island + clouds
roi_buoy_initial = [6.4325e+02, 5.0e+02, 35.5, 35.5]; % Initial ROI surrounding buoy; based on first frame
roi_buoy_tracker = 31;                                % Size of scanning ROI-region surrounding the KLT-tracker
mThreshold = 1300;                                    % Strictness of feature extraction for stabilization transforms

% Initialize other variables
trackerAlive = 0;     % State of KLT-tracker (1)
trackerWasAlive = 0;  % State of KLT-tracker (2)
KLT_roi_size = 5;     % Block size of KLT-tracker
KLT_points = [];      % Points spit out by KLT-tracker
ii = 2;               % Loop variable for frames
Hcumulative = eye(3); % Initial transformation matrix (stabilization)

% Create video viewer
hVPlayer = vision.VideoPlayer; 

% Create tracker object
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'BlockSize', [KLT_roi_size KLT_roi_size]);

% Initialize frames
imgB = rgb2gray(im2single(readFrame(hVideoSrc)));
imgBp = imgB;

% Find features of interest in first frame
points = detectBRISKFeatures(imgBp, 'MinQuality', 0.6, 'MinContrast', 0.3, 'ROI', roi_buoy_initial);

% ALGORITHM
% 1. Initialize ROI
% 2. Wait till FeatureFinder detects point in ROI. If yes, initialize KLT-object
% 3. Let KLT-tracker do its own thing
% 4. If no point has been returned by KLT-tracker (or point is invalid), the buoy is probably missing. Disable KLT-object
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
    
    % Save warped image as a frame
    frame = imgBp;

    % Check if buoy is detected in ROI using FeatureFinder
    if (trackerAlive == 0) % Tracker not active

        % Detect features using BRISK-algorithm 
        if (trackerWasAlive == 0)     % Use initial knowledge of ROI around buoy to start tracking it [STRICT]
            points = detectBRISKFeatures(frame, 'MinQuality', 0.3, 'MinContrast', 0.3, 'ROI', roi_buoy_initial);
        elseif (trackerWasAlive == 1) % Use previous knowledge of KLT-tracker point to cast new ROI around it [WEAK](
            points = detectBRISKFeatures(frame, 'MinQuality', 0.3, 'MinContrast', 0.2, 'ROI', [KLT_points(1) - floor(roi_buoy_tracker/2), KLT_points(2) - floor(roi_buoy_tracker/2), roi_buoy_tracker, roi_buoy_tracker]);
        end

        % Check if points were returned by FeatureFinder in ROI
        if ~isempty(points) 
            % Initialize the KLT-tracker on the found point
            initialize(tracker, points.Location, frame);
            
            % Return points of the KLT-tracker
            [KLT_points, validity] = tracker(frame);
    
            % Insert marker in frame to indicate (potential) buoy
            frame = insertMarker(frame, KLT_points(validity, :), '+', 'Color', 'white');
            frame = insertShape(frame, 'Rectangle', [KLT_points(1) - floor(KLT_roi_size/2), ...
                KLT_points(2) - floor(KLT_roi_size/2), KLT_roi_size, KLT_roi_size], 'Color', 'white');
    
            % Indicate that tracker is alive
            trackerAlive = 1;
            trackerWasAlive = 1;
        elseif (isempty(points) && trackerWasAlive == 1) 
            % Insert marker in frame to indicate ROI
            frame = insertShape(frame, 'Rectangle', [KLT_points(1) - floor(roi_buoy_tracker/2), ...
                KLT_points(2) - floor(roi_buoy_tracker/2), roi_buoy_tracker, roi_buoy_tracker], 'Color', 'white');
        end

    elseif (trackerAlive == 1) % Tracker active

        % DON'T INITIALIZE TRACKER AGAIN
        % LET IT DO ITS THING
        
        % Return points of the KLT-tracker
        [KLT_points, validity] = tracker(frame);

        % Check ROI based on current returned point of KLT-tracker [WEAK]
        points = detectBRISKFeatures(frame, 'MinQuality', 0.2, 'MinContrast', 0.2, 'ROI', [KLT_points(1) - roi_buoy_tracker, KLT_points(2) - roi_buoy_tracker, (2*roi_buoy_tracker + 1), (2*roi_buoy_tracker + 1)]);
        frame = insertShape(frame, 'Rectangle', [KLT_points(1) - floor(roi_buoy_tracker/2), KLT_points(2) - floor(roi_buoy_tracker/2), roi_buoy_tracker, roi_buoy_tracker], 'Color', 'white');

        % Are there no points returned or KLT points are no longer valid? 
        % --> tracker should not be alive anymore
        if (isempty(KLT_points) || isempty(points) || (validity == 0))
            release(tracker); % Release the tracker
            trackerAlive = 0; % Disable tracker
        else
            % Insert marker in frame to indicate (potential) buoy
            frame = insertMarker(frame, KLT_points(validity, :), '+', 'Color', 'white');
            frame = insertShape(frame, 'Rectangle', [KLT_points(1) - floor(KLT_roi_size/2), ...
                KLT_points(2) - floor(KLT_roi_size/2), KLT_roi_size, KLT_roi_size], 'Color', 'white');
        end
    end

    % Convert frame to correct dimensions (dirty fix)
    if (size(frame, 3) == 3)
        frame = rgb2gray(im2single(frame));
    end

    % Play current frame in video player
    step(hVPlayer, frame);

    % Increment frame counter
    ii = ii + 1;
end

% Release video viewer
release(hVPlayer);