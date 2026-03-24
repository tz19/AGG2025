clear all; clc; close all;

P = pathConfig;

% ----- Same as main.m: number of orientation families (GI IDs 0 .. nGF-1) -----
nGF = 16;

% ----- Time series: GI file index range -----
stepFirst = 0;
stepLast = 95;
steps = stepFirst:stepLast;
nSteps = numel(steps);

% ----- Physical pixel size (square pixel edge length) -----
pixel_nm = 42.5;
pixel_um = pixel_nm * 1e-3;

% ----- Data root (switch to P.rootW100, P.rootCr100, etc.; Cr may use P.giCrCsv) -----
rootData = P.rootNi100;
giPath = @(k) P.giCsv(rootData, k);

time = zeros(nSteps, 1);
totalGrainCount = zeros(nSteps, 1);
grainCountByFamily = zeros(nSteps, nGF);
areaFracByFamily = zeros(nSteps, nGF);
avgSizeByFamily = zeros(nSteps, nGF);
% detectGrains avgGrainSize = mean grain area in pixel count (px^2)
avgGrainAreaPxGlobal = zeros(nSteps, 1);

parfor j = 1:nSteps
    k = steps(j);
    [time(j), ~, GI] = readCSVfile(giPath(k));
    [numGrains, areaRatio, avgGrainSize] = detectGrains(nGF, GI);
    grainCountByFamily(j, :) = numGrains.';
    areaFracByFamily(j, :) = areaRatio.';
    avgSizeByFamily(j, :) = avgGrainSize.';
    tg = sum(numGrains);
    if tg > 0
        avgGrainAreaPxGlobal(j) = sum(numGrains .* avgGrainSize) / tg;
    else
        avgGrainAreaPxGlobal(j) = NaN;
    end
    totalGrainCount(j) = tg;
end

% Mean area (px^2) -> length scale: sqrt(area) * pixel edge -> micrometers
avgGrainLength_um = pixel_um * sqrt(avgGrainAreaPxGlobal);

% Same scaling as trackGrain: map CSV time to seconds
time_s = (time - 5) * 45;
idx = 6:3:nSteps;

close all;

% ----- Fig 1: total grain count -----
fig1 = figure(1);
plot(time_s(idx), totalGrainCount(idx), '-o', 'Color', 'red', 'LineWidth', 1.8, 'MarkerSize', 5);
grid on;
box on;
set(gca, 'FontSize', 16, 'LineWidth', 1.2, 'TickDir', 'out');
xlim([-100, 2100]);
ylim([100, 900]);
xticks(0:500:2000);
yticks(100:200:900);
xlabel('$t$ (s)', 'Interpreter', 'latex', 'FontSize', 22);
ylabel('Number of grains', 'Interpreter', 'latex', 'FontSize', 22);
% copygraphics(fig1, "BackgroundColor", "none", "Resolution", 300)

% ----- Fig 2: global mean grain length sqrt(<A>_px) * pixel edge (µm) -----
fig2 = figure(2);
plot(time_s(idx), avgGrainLength_um(idx), '-o', 'Color', 'red', ...
    'LineWidth', 1.8, 'MarkerSize', 5);
grid on;
box on;
set(gca, 'FontSize', 16, 'LineWidth', 1.2, 'TickDir', 'out');
xlim([-100, 2100]);
ylim([0.5,1.25])
xticks(0:500:2000);
yticks(0.0:0.1:1.3)
ys = avgGrainLength_um(idx);
xlabel('$t$ (s)', 'Interpreter', 'latex', 'FontSize', 22);
ylabel('Average grain size ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 22);
% copygraphics(fig2, "BackgroundColor", "none", "Resolution", 300)

% ----- Fig 3: two orientation families, area fraction / initial (example style) -----
A0 = areaFracByFamily(1, :);
normArea = nan(nSteps, nGF);
for f = 1:nGF
    if A0(f) > 0
        normArea(:, f) = areaFracByFamily(:, f) / A0(f);
    end
end

fam0 = 1;      % theta = 0 deg
fam45 = nGF;   % theta = 45 deg

fig3 = figure(3);
hold on;
plot(time_s(idx), normArea(idx, fam0), '-o', ...
    'Color', [0.65 0.00 0.00], 'LineWidth', 1.8, 'MarkerSize', 6, ...
    'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.65 0.00 0.00]);
plot(time_s(idx), normArea(idx, fam45), '-o', ...
    'Color', [0.00 0.10 0.85], 'LineWidth', 1.8, 'MarkerSize', 6, ...
    'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.00 0.10 0.85]);
hold off;
grid on;
box on;
set(gca, 'FontSize', 16, 'LineWidth', 1.2, 'TickDir', 'out');
xlim([-100, 2100]);
xticks(0:500:2000);
xlabel('$t$ (s)', 'Interpreter', 'latex', 'FontSize', 22);
ylabel('$A/A_0$', 'Interpreter', 'latex', 'FontSize', 22);
legend({'$\theta=0^\circ$ - [100]', '$\theta=45^\circ$ - [110]'}, ...
    'Location', 'northwest', 'Box', 'on', 'Interpreter', 'latex', 'FontSize', 14);
copygraphics(fig3, "BackgroundColor", "none", "Resolution", 300)
