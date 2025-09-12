clear all; clc; close all
%% Material Parameters

% Single crystal elastic constants for Cu, Ni, W, Cr
C11s = [168.4E9 , 246.5E9, 522.4E9, 339.8E9];    %(Pa)
C12s = [121.4E9 , 147.3E9, 204.4E9, 58.6E9];    %(Pa)
C44s = [75.4E9  , 124.7E9, 160.8E9, 99.0E9];    %(Pa)
titlestr = {'Cu ($A = 3.21$)','Ni ($A = 2.51$)', 'W ($A = 1.01$)','Cr ($A = 0.70$)'};

a = 4; % 1: Cu, 2: Ni, 3: W, 4: Cr

C11 = C11s(a);
C12 = C12s(a);
C44 = C44s(a);
K_bar = (C11+2*C12)/3;
G_roots = roots([8 5*C11+4*C12 -C44*(7*C11-4*C12) -C44*(C11-C12)*(C11+2*C12)]);
mu_bar = G_roots(2);

theta = 0:45;
index = 1:3:max(theta)+1;
E = zeros(size(theta));
n = [1,0,0];

for i = 1:length(theta)

    % eul = [45, rad2deg(acos(1/sqrt(3))), theta(i)];  % in degree
    eul = [theta(i),0 ,0];
    Q = zxz_rotation_matrix(eul); % e_global = Q * e_local
    E(i) = diffractionElasticConst2D(n,Q,C11,C12,C44)/1e9;

end

plot(theta,E,'-r',LineWidth=1.5,HandleVisibility='off')
hold on
plot(theta(index),E(index),'or',LineWidth=2.0)
grid on
xlim([0,max(theta)])
ylim([260,310])
set(gca, "FontSize", 18, 'LineWidth',1.0, 'TickDir','out')
ax = gca;
ax.XTick = 0:9:max(theta);
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = 0:3:45;

ax.YTick = 260:10:310;
ax.YAxis.MinorTick = "on";
ax.YAxis.MinorTickValues = 260:2.5:310;
xlabel('$\theta_{3}$ ($^\circ$)', 'Interpreter', 'latex','FontSize',22);
ylabel('$\widetilde{E}$ (GPa)', 'Interpreter', 'latex','FontSize',22)
% title(titlestr{a},Interpreter='latex')

copygraphics(gca,'Resolution',300,'BackgroundColor','none')