clear all; clc; close all;

P = pathConfig;

% x = 220;
% y = 95;

x = 206;
y = 81;

time = zeros(96,1);
Area_Ni = zeros(96,1);
Area_W = zeros(96,1);
Area_Cr = zeros(96,1);

% Initial
filePath = P.giCsv(P.rootNi100, 0);
[~, ~, GI] = readCSVfile(filePath);
target_gid = GI(round(y), round(x));
[area, centroid, distance] = getClosestGrain(GI, target_gid, x, y);
x = centroid(1);
y = centroid(2);
disp(['Grain ID: ' num2str(target_gid)])
disp(['Centroid: ' num2str(centroid)])

rootW = P.rootW100;
rootNi = P.rootNi100;
rootCr = P.rootCr100;
parfor i = 1:65

    % Ni: same case as t=0 initial GI; time axis from CSV column 1
    filePath_Ni = fullfile(rootNi, 'output', 'GrainIndeics', 'GrainIndex', sprintf('GI_%d.csv', i-1));
    [time(i), ~, GI_Ni] = readCSVfile(filePath_Ni);
    [area, ~, distance] = getClosestGrain(GI_Ni, target_gid, x, y);
    if distance < 2*sqrt(area)
        Area_Ni(i) = area;
    end

    % W
    filePath_W = fullfile(rootW, 'output', 'GrainIndeics', 'GrainIndex', sprintf('GI_%d.csv', i-1));
    [~, ~, GI_W] = readCSVfile(filePath_W);
    [area, ~, distance] = getClosestGrain(GI_W, target_gid, x, y);
    if distance < 2*sqrt(area)
        Area_W(i) = area; 
    end

    % Cr
    filePath_Cr = fullfile(rootCr, 'output', 'GrainIndeics', 'GrainIndex', sprintf('GI_%d.csv', i-1));
    [~, ~, GI_Cr] = readCSVfile(filePath_Cr);
    [area, ~, distance] = getClosestGrain(GI_Cr, target_gid, x, y);
    if distance < 2*sqrt(area)
        Area_Cr(i) = area;
    end

end



fig = figure(3);
hold on

plot((time(6:end)-5)*45, Area_W(6:end)/Area_W(6),'k-',LineWidth=2)
plot((time(6:end)-5)*45, Area_Cr(6:end)/Area_Cr(6),'b-',LineWidth=2)
plot((time(6:end)-5)*45, Area_Ni(6:end)/Area_Ni(6),'r-',LineWidth=2)
set(gca,'Box','on', "FontSize", 22, 'LineWidth',1.5, 'TickDir','out')
xlabel('$t$ (s)','Interpreter','latex','FontSize',28);
ylabel('$A/A_{0}$', 'Interpreter', 'latex', 'FontSize', 28);
xlim([0,45]*45)
ylim([0.2,1.4])
% ylim([0,5])
xticks(0:450:2025)
yticks(0.2:0.2:1.4)
% legend('W','Cr','Ni','Location','southwest','Box','off','FontSize', 24)
set(fig, 'Position', [100, 100, 1100, 600])
copygraphics(gca,"BackgroundColor","none","Resolution",300)
hold off