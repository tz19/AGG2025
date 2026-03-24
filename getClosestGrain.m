function [area, centroid, distance] = getClosestGrain(GI, target_gid, ref_x, ref_y)
% GETCLOSESTGRAIN Find the target-grain region closest to a reference point and report its stats.
% Inputs:
%   GI        - Grain-index map (2D)
%   target_gid- Target grain ID (integer label in GI)
%   ref_x     - Reference x (0-based grid index)
%   ref_y     - Reference y (0-based grid index)
% Outputs:
%   area      - Grain area (pixel count)
%   centroid  - Centroid [x, y] in continuous coordinates (1-based consistent with regionprops)

% Binary mask and connected components
binaryMap = GI == target_gid;
CC = bwconncomp(binaryMap, 4);

% No pixel with target_gid
if CC.NumObjects == 0
    area = NaN;
    centroid = [NaN, NaN];
    distance = 1000;
    return;
end

% Geometry per connected region
stats = regionprops(CC, 'Centroid', 'Area');
centroids = vertcat(stats.Centroid);

ref_point = [ref_x, ref_y];

% Distance from each centroid to the reference point; pick nearest region
distances = pdist2(ref_point, centroids);
[~, min_idx] = min(distances);

% Pack outputs
area = stats(min_idx).Area;
centroid = stats(min_idx).Centroid;
distance = norm(centroid - ref_point, 2);

end