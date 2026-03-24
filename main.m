clear all; clc; close all;

P = pathConfig;
Num = 25;

filePath_GI = P.giCsv(P.rootW100, Num);
% filePath_GI = P.giCsv(P.rootNi100, Num);
% filePath_GI = P.giCrCsv(P.rootCr100, Num);

zTolerance = 1e-8;
zValueToKeep = 1;
x_col = 3; y_col = 4; z_col = 5; gi_col = 2;

% Load GI field
[time, maxGI, GI] = readCSVfile(filePath_GI);

fig_GI = plotOrientationMap(imresize(GI,2,"nearest"),'Verbose', false,'ShowColorbar', false);
exportgraphics(fig_GI, 'output.png','Resolution',600,'BackgroundColor','none')
% copygraphics(fig_GI,"BackgroundColor","none","Resolution",300)


nGF = 16;

% Analyze each orientation family
% [numGrains, areaRatio, avgGrainSize, avgValue] = detectGrains(nGF, GI, 'ValueMatrix', ValueMatrix);
[numGrains, areaRatio, avgGrainSize] = detectGrains(nGF, GI);

% === Step 3: Compute Average Size Metric ===
avgSize = sqrt(sum(areaRatio .* avgGrainSize));

% Show summary
disp(['Number of grains per family: ', num2str(numGrains')]);
disp(['Average grain size per family: ', num2str(avgGrainSize')]);
disp(['Area ratio of each family: ', num2str(areaRatio')]);

figure(2)
map = flipud(jet(16));
G0 = bar(linspace(0,45,nGF),areaRatio,0.6,'EdgeColor','k','LineWidth',1);

G0.FaceColor = "flat";
for i = 1:nGF
    G0.CData(i,:) = map(i,:);
end
ylim([0,0.25])
yticks(0:0.05:0.25)
xticks(0:9:45)

ax = gca;
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = 0:3:45;
ax.YAxis.MinorTick = "on";
ax.YAxis.MinorTickValues = 0:0.025:0.25;

set(gca, "FontSize", 18, 'LineWidth',1.5, 'TickDir','out')

hXlabel = xlabel('$\theta_{3}$ ($^\circ$)','Interpreter','latex','FontSize',24);
hYlabel = ylabel('Area fraction', 'Interpreter', 'latex', 'FontSize', 24);

copygraphics(gca,"BackgroundColor","none","Resolution",300)
            