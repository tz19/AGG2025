clear all; clc; close all;

P = pathConfig;
simple_point = 5:4:96;

time = zeros(size(simple_point));
energy = zeros(length(simple_point),3);

rootNi = P.rootW100;
parfor i = 1:length(simple_point)

    Num = simple_point(i);
    % bulk free energy
    filePath_F0 = fullfile(rootNi, 'output', 'F0', sprintf('FE_%d.csv', Num));
    [time(i), ~, F0] = readCSVfile(filePath_F0);
    meanF0 = mean(2*F0(:));
    % Gradient energy
    filePath_F1 = fullfile(rootNi, 'output', 'F1', sprintf('GE_%d.csv', Num));
    [~, ~, F1] = readCSVfile(filePath_F1);
    meanF1 = mean(1.0*F1(:));
    % Elastic energy
    filePath_F2 = fullfile(rootNi, 'output', 'F2', sprintf('WE_%d.csv', Num));
    [~, ~, F2] = readCSVfile(filePath_F2);
    meanF2 = mean(F2(:));

    energy(i,:) = [meanF0, meanF1, meanF2];

end
time = time - 5;
energy = energy/sum(energy(1,1:2));

save(P.energyMat, "energy")

figure(1)
hold on

% choose color
map = cool;
num = size(energy,2); % number of series (columns)
idx = linspace(1,128,num);
idx = round(idx);
C = map(idx,:);

G0 =bar(time,energy,0.5,'stacked','EdgeColor','k','LineWidth',0.5);
for i = 1:3
    G0(i).FaceColor = C(i,:);
end
xlabel('$t$ (s)','Interpreter','latex','FontSize',18);
ylabel('$E/E_{0}$', 'Interpreter', 'latex', 'FontSize', 18);
legend('Bulk Free Energy', 'Gradient Energy', 'Elastic energy', 'Location','northeast','Box', 'off', 'NumColumns', 1)
% ylim([0,1.4])
% yticks(0:0.2:1.4)
xticks(0:6:42)
set(gca, 'Box', 'on', "FontSize", 14, 'LineWidth',1.0, 'TickDir','out')
hold off

copygraphics(gca,"BackgroundColor","none","Resolution",300)