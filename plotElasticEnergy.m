clear all; clc; close all;

P = pathConfig;
Num = 55;

filePath_GI = P.f2We(P.rootNi100, Num);

zTolerance = 1e-8;
zValueToKeep = 1;
x_col = 3; y_col = 4; z_col = 5; gi_col = 2;

[time, maxGI, WE] = readCSVfile(filePath_GI);

data = WE;  % replace with your own field if needed

% Axes (0..400 grid nodes)
x = 0:400;    % row index -> x
y = 0:400;    % column index -> y

% Grid (order: meshgrid(y, x))
[X, Y] = meshgrid(y, x);

% Fixed colormap limits (same idea as orientation map; stable across exports)
climLow = 0.2;
climHigh = 0.7;
contourLevels = linspace(climLow, climHigh, 256);

% Plot
fig = figure;
[C, h] = contourf(X, Y, data, contourLevels, 'LineColor', 'none');

% Interpolated shading + colormap
shading interp;
colormap('turbo');

% Apply limits (no colorbar; mapping only)
clim([climLow, climHigh]);

% Equal aspect ratio
axis equal;

% Match plotOrientationMap (Verbose=false): no box, no ticks or numeric labels
set(gca, ...
    'YDir', 'normal', ...
    'Box', 'off', ...
    'TickLength', [0 0], ...
    'XTick', [], ...
    'YTick', [], ...
    'XTickLabel', '', ...
    'YTickLabel', '');

% copygraphics(fig,"BackgroundColor","none","Resolution",600)
exportgraphics(fig, P.welsPng, 'Resolution', 600, 'BackgroundColor', 'white')