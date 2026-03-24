function varargout = detectGrains(nGF, GI, varargin)
% detectGrains - Analyze connected components of grains and optionally compute average values
%
% Usage:
% [numGrains, areaRatio, avgGrainSize] = detectGrains(nGF, GI)
% [numGrains, areaRatio, avgGrainSize, avgValue] = detectGrains(nGF, GI, 'ValueMatrix', ValueMatrix)
%
% Inputs:
%   nGF           - Number of time frames or grain fields to analyze
%   GI            - Grain index matrix (2D), where each value represents a grain ID
%   varargin      - Optional name-value pairs
%
% Optional Parameters:
%   'ValueMatrix' - A numeric matrix of the same size as GI for computing average values per grain
%
% Outputs:
%   numGrains     - Number of grains at each frame/time step
%   areaRatio     - Ratio of total grain area over total grid area
%   avgGrainSize  - Average size (pixel count) of grains
%   avgValue      - Average value in grain regions (if ValueMatrix is provided)

% === Parse optional inputs ===
p = inputParser;
addParameter(p, 'ValueMatrix', [], @isnumeric);  % Optional parameter
parse(p, varargin{:});

hasValue = ~isempty(p.Results.ValueMatrix);
ValueMatrix = p.Results.ValueMatrix;

% === Initialize output arrays ===
numGrains = zeros(nGF, 1);
areaRatio = zeros(nGF, 1);
avgGrainSize = zeros(nGF, 1);
avgValue = NaN(nGF, 1);  % Only used if ValueMatrix is provided

totalCells = size(GI, 1) * size(GI, 2);

% === Main loop over frames or grain fields ===
for i = 1:nGF
    currentID = i - 1;
    binaryMap = (GI == currentID);
    
    % Skip if no such grain exists in this frame
    if ~any(binaryMap(:))
        continue;
    end
    
    CC = bwconncomp(binaryMap, 4);
    numGrains(i) = CC.NumObjects;

    pixelIdxList = CC.PixelIdxList;
    areas = cellfun(@length, pixelIdxList);
    avgGrainSize(i) = mean(areas);
    areaRatio(i) = sum(areas) / totalCells;

    % Compute average value within grain regions if provided
    if hasValue
        valuesInRegions = cellfun(@(idx) ValueMatrix(idx), pixelIdxList, 'UniformOutput', false);
        allValues = vertcat(valuesInRegions{:});
        avgValue(i) = mean(allValues);
    end
end

% === Set output arguments based on requested number and provided inputs ===
switch nargout
    case 3
        varargout{1} = numGrains;
        varargout{2} = areaRatio;
        varargout{3} = avgGrainSize;
    case 4
        if hasValue
            varargout{1} = numGrains;
            varargout{2} = areaRatio;
            varargout{3} = avgGrainSize;
            varargout{4} = avgValue;
        else
            error('Cannot return 4 outputs: ValueMatrix not provided.');
        end
    otherwise
        error('Unsupported number of output arguments or invalid input combination.');
end