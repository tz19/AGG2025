function [time, maxValue, Value] = readCSVfile(filePath, varargin)
% readCSVfile - Reads a CSV file and constructs a matrix based on spatial coordinates.
% Only rows with z ≈ zTarget are kept.
%
% Usage:
%   [time, maxValue, Value] = readCSVfile(filePath, ... name-value pairs ...)
%   Column 1 of the CSV is time (one scalar per file, repeated on each row); returned as time.
%
% Optional parameters with defaults:
%   x_col       - column index for x-coordinate
%   y_col       - column index for y-coordinate
%   z_col       - column index for z-coordinate
%   value_col   - column index for value to store in matrix
%   zTarget     - only keep rows where z ≈ zTarget
%   tolerance   - tolerance for z filtering
%   gridSize    - size of the output matrix [m, n]

% === Defaults and input parsing ===
p = inputParser;
addParameter(p, 'x_col', 3, @isnumeric);
addParameter(p, 'y_col', 4, @isnumeric);
addParameter(p, 'z_col', 5, @isnumeric);
addParameter(p, 'value_col', 2, @isnumeric);
addParameter(p, 'zTarget', 1, @isnumeric);
addParameter(p, 'tolerance', 1e-8, @isnumeric);
addParameter(p, 'gridSize', [401, 401], @(x) isvector(x) && length(x)==2);

parse(p, varargin{:});

% Parsed parameters
x_col = p.Results.x_col;
y_col = p.Results.y_col;
z_col = p.Results.z_col;
value_col = p.Results.value_col;
zTarget = p.Results.zTarget;
tolerance = p.Results.tolerance;
gridSize = p.Results.gridSize;

% === Load table ===
temp = readmatrix(filePath, "NumHeaderLines", 1);
if isempty(temp)
    error('Failed to read data from the CSV file.');
end

% Column 1 is time (same for all rows in one snapshot)
time = temp(1, 1);

% === Rows matching zTarget ===
selectedRows = abs(temp(:, z_col) - zTarget) < tolerance;
result = temp(selectedRows, :);

% === x, y, value columns ===
if isempty(value_col)
    error('value_col must be specified.');
end

x = result(:, x_col);
y = result(:, y_col);
value_v = result(:, value_col);

% === Drop out-of-range coordinates ===
m = gridSize(1); n = gridSize(2);
valid = (x >= 0) & (x <= m-1) & (y >= 0) & (y <= n-1);
x = x(valid); y = y(valid); value_v = value_v(valid);

% === Allocate output matrix ===
Value = zeros(m, n);

% === Assign (MATLAB uses 1-based indexing) ===
if ~isempty(x)
    indices = sub2ind([m, n], x + 1, y + 1);
    Value(indices) = value_v;
end

maxValue = max(value_v(:));

end