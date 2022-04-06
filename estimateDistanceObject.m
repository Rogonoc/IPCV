function D = estimateDistanceObject(cameraParams, pointOfInterest, imagePoints, worldPoints, z_centreOfWorld)
% D = estimateDistanceObject(cameraParams, pointOfInterest, imagePoints, worldPoints, z_centreOfWorld)
%  Estimates the distance between a point of interest in an image by 
%  using the knowledge between the relation of points in the image and
%  their real distances, preferably given in a checker-board pattern.
%
%  cameraParams: the intrinsic parameters of the calibrated camera,
%       Uses cameraParameters-object
%
%  pointOfInterest: the point for which the distance needs to be estimated
%       Vector of form [x y]
%
%  imagePoints: defined image coordinates of points in an image
%       Vector of form M-by-2
%
%  worldPoints: the real-world relation (given in [m]) between these points
%       Vector of form M-by-2
%
%  z_centreOfWorld: the prior knowledge of the centre of The world (given in [m]), which 
%                   is the camera height above a known level; default value is z = 0 [m]
%
% BASED ON: The section "Measure the Distance to The First Coin", showcased in the 
%           tutorial "Measuring Planar Objects with a Calibrated Camera" from MATLAB ©

% Set default parameters
if nargin < 5 || isempty(z_centreOfWorld)
    z_centreOfWorld = 0; % Assumed centre of world lies at z = 0
end

%% Compute the location of the centre of the world
% Compute the extrinsic parameters, so the rotation and translation of the camera
[R, t] = extrinsics(imagePoints, worldPoints, cameraParams);
    
% Convert this position to world coordinates
center_world = pointsToWorld(cameraParams, R, t, pointOfInterest); % Take mean of point of interests

% Add z-coordinate
center_world = [center_world z_centreOfWorld];

%% Compute the distance to the camera based on the extrinsics
[~, cameraLocation] = extrinsicsToCameraPose(R, t);

%% Compute the distance between point of interest and camera 
D = norm(center_world - cameraLocation); % in [m]
%fprintf('Distance from the camera to the buoy = %0.2f m\n', D);