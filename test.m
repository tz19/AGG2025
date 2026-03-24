clear all; clc; close all;

P = pathConfig;
load(P.energyMat)

G0 =bar(energy,0.5,'stacked','EdgeColor','k','LineWidth',0.5);

ax = gca;
ax.XAxisLocation = "top";
ax.XTick = [];
% ax.XAxis.Visible = 'off';


hold on


% 
% set(gca,'Position',[0.06,0.06,.92,.92]);
% prettyAxes().gbase()
% 
% [ax, ax2] = truncAxis(G0, 'Y',[1.2,14]);
% 
% 
hold off




