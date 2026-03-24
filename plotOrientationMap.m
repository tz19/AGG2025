function fig = plotOrientationMap(orientationMatrix, varargin)
% plotOrientationMap Identifies grains, extracts grain boundaries, and plots the orientation map.
%
%   fig = plotOrientationMap(orientationMatrix)
%   fig = plotOrientationMap(orientationMatrix, 'ShowLabeledGrains', true/false)
%   fig = plotOrientationMap(orientationMatrix, 'PlotRegion', [x, y, w, h])
%   fig = plotOrientationMap(orientationMatrix, 'DrawBoundaries', true/false)
%   fig = plotOrientationMap(orientationMatrix, 'OverlayBoundaryLines', true/false)
%   fig = plotOrientationMap(orientationMatrix, 'Verbose', true/false)
%   fig = plotOrientationMap(orientationMatrix, 'ShowColorbar', true/false)
%
%   Inputs:
%       orientationMatrix - A 2D matrix where each element represents crystallographic orientation (integer).
%       varargin          - Optional name-value pairs:
%                           'ShowLabeledGrains'        - Boolean, whether to display labeled grains image (default false)
%                           'PlotRegion'               - [x, y, w, h], restrict plotting region (default is full image)
%                           'DrawBoundaries'           - Boolean, mark boundary pixels as black (default true)
%                           'OverlayBoundaryLines'     - Boolean, overlay boundary lines on image (default false)
%                           'Verbose'                  - Boolean, show progress messages (default true)
%                           'ShowColorbar'             - Boolean, show color bar
%   Outputs:
%       fig - Handle to the generated figure.

% --- Parameter parsing ---
p = inputParser;
addRequired(p, 'orientationMatrix', @(x) isnumeric(x) && ismatrix(x) && ~isempty(x));
addParameter(p, 'ShowLabeledGrains', false, @islogical);
addParameter(p, 'PlotRegion', [], @(x) isempty(x) || (isvector(x) && length(x) == 4 && all(x > 0)));
addParameter(p, 'DrawBoundaries', true, @islogical); % Default: draw boundaries
addParameter(p, 'OverlayBoundaryLines', false, @islogical);
addParameter(p, 'Verbose', true, @islogical);
addParameter(p, 'ShowColorbar', true, @islogical); % Default: no colorbar
parse(p, orientationMatrix, varargin{:});

originalOrientationMatrix = p.Results.orientationMatrix;
showLabeledGrainsFig = p.Results.ShowLabeledGrains;
plotRegion = p.Results.PlotRegion;
drawBoundaries = p.Results.DrawBoundaries;
overlayBoundaryLines = p.Results.OverlayBoundaryLines;
verbose = p.Results.Verbose;
showColorbar = p.Results.ShowColorbar;

[rows, cols] = size(originalOrientationMatrix);

% Parse plot region
if isempty(plotRegion)
    xRange = [1, cols];
    yRange = [1, rows];
else
    xStart = max(1, round(plotRegion(1)));
    yStart = max(1, round(plotRegion(2)));
    width  = max(1, round(plotRegion(3)));
    height = max(1, round(plotRegion(4)));

    xEnd = min(cols, xStart + width - 1);
    yEnd = min(rows, yStart + height - 1);

    xRange = [xStart, xEnd];
    yRange = [yStart, yEnd];
end

% --- Grain labeling ---
if verbose
    disp('Step 1: Unique grain labeling...');
end

uniqueGrainLabels = zeros(rows, cols);
currentMaxLabel = 0;

existingOrientations = unique(originalOrientationMatrix(:))';

for k = existingOrientations
    binaryMask = (originalOrientationMatrix == k);
    CC = bwconncomp(binaryMask, 4);
    for i = 1:CC.NumObjects
        currentMaxLabel = currentMaxLabel + 1;
        uniqueGrainLabels(CC.PixelIdxList{i}) = currentMaxLabel;
    end
end

numberOfUniqueGrains = currentMaxLabel;

if verbose
    disp(['Step 1 completed. Found ', num2str(numberOfUniqueGrains), ' unique grains.']);
end

% Display labeled grains if requested
if showLabeledGrainsFig && numberOfUniqueGrains > 0
    figure;
    imagesc(uniqueGrainLabels);
    colormap(lines(min(numberOfUniqueGrains + 1, 256)));
    axis image;
    title('Unique Labeled Grains');
    colorbar;
    if verbose
        disp('Displayed labeled grains.');
    end
end

if numberOfUniqueGrains == 0
    error('No grains were identified in the matrix. Cannot proceed with plotting.');
end

% Extract grain boundaries only if needed
allBoundaryPixelIndices = [];
grainBoundaries = cell(numberOfUniqueGrains, 1);

if drawBoundaries || overlayBoundaryLines
    if verbose
        disp('Step 2: Extracting grain boundaries...');
    end

    allBoundaryPixelIndices = [];
    for i = 1:numberOfUniqueGrains
        grainMask = (uniqueGrainLabels == i);
        B = bwboundaries(grainMask, 'noholes');
        if ~isempty(B)
            boundaryCoords = B{1};
            grainBoundaries{i} = boundaryCoords;
            idx = sub2ind(size(originalOrientationMatrix), boundaryCoords(:,1), boundaryCoords(:,2));
            allBoundaryPixelIndices = [allBoundaryPixelIndices; idx];
        else
            grainBoundaries{i} = [];
        end
    end
    allBoundaryPixelIndices = unique(allBoundaryPixelIndices);
end

% Build display matrix and mark boundaries if required
displayMatrix = double(originalOrientationMatrix);
minOrientationVal = 0;
boundaryMarkerValue = minOrientationVal - 1;

if drawBoundaries && ~isempty(allBoundaryPixelIndices)
    displayMatrix(allBoundaryPixelIndices) = boundaryMarkerValue;
end

% --- Final plotting ---
if verbose
    disp('Step 3: Plotting orientation map...');
end

% fig = figure('Position',[100, 100, 1000, 750]);
fig = figure;
hold on;


% Determine color map based on number of orientations
uniqueActualOrientations = sort(unique(originalOrientationMatrix(:)));
numActualOrientations = length(uniqueGrainLabels);
if numActualOrientations == 0
    grain_cmap = [0 0 1];
elseif numActualOrientations == 1
    grain_cmap = flipud(jet(1));
else
    grain_cmap = flipud(jet(16));
end

final_colormap = [0 0 0; grain_cmap]; % First row is black for boundaries
imagesc(displayMatrix);
colormap(final_colormap);
axis image;

% Set color limits
clim([boundaryMarkerValue, 15]);

% Optionally display colorbar
if showColorbar && ~isempty(uniqueActualOrientations)
    cb = colorbar;
    cb.Ticks = uniqueActualOrientations;
    cb.TickLabels = arrayfun(@num2str, uniqueActualOrientations, 'UniformOutput', false);
end

% Set axis limits to restrict plot region
xlim(xRange);
ylim(yRange);

% Customize axes appearance
if ~verbose
    set(gca, ...
        'YDir', 'normal', ...
        'TickLength', [0 0], ...
        'XTickLabel', '', ...
        'YTickLabel', '', ...
        'Box', 'off');
end

if isempty(plotRegion)
    rectangle('Position', [xRange(1), yRange(1), xRange(2)-xRange(1), yRange(2)-yRange(1)], ...
        'EdgeColor', 'k', 'FaceColor', 'none', 'LineWidth', 1.0);
end

% Overlay boundary lines if requested
if overlayBoundaryLines
    fullWidth = cols;
    fullHeight = rows;
    plotWidth = xRange(2) - xRange(1) + 1;
    plotHeight = yRange(2) - yRange(1) + 1;
    scaleFactor = min(fullWidth / plotWidth, fullHeight / plotHeight);
    lineWidth = max(1.0, scaleFactor / 2);  % Avoid excessive thickness

    for i = 1:numberOfUniqueGrains
        coords = grainBoundaries{i};
        if ~isempty(coords)
            plot(coords(:,2), coords(:,1), 'k-', 'LineWidth', lineWidth);
        end
    end
end

hold off;

if verbose
    disp('Plotting completed.');
end

end