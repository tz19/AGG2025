clear all; clc; close all
%% Material Parameters

% Single crystal elastic constants for Cu, Ni, W, Cr
C11s = [168.4E9 , 246.5E9, 522.4E9, 339.8E9];    %(Pa)
C12s = [121.4E9 , 147.3E9, 204.4E9, 58.6E9];    %(Pa)
C44s = [75.4E9  , 124.7E9, 160.8E9, 99.0E9];    %(Pa)

a = 2; % Ni

C11 = C11s(a);
C12 = C12s(a);
C44 = C44s(a);

TriGamma = linspace(0, 1, 100);

Gamma = TriGamma / 3;

[a,b,c] = U(C11,C12,C44);

S = (3*a+4*b)/3 - 4*(b-c)*Gamma;

E = 1./S;

plot(TriGamma,E/1e9,'-r',LineWidth=2.0,HandleVisibility='off')
hold on
tempx = [0, 19/121, 1/4, 1/3]' * 3;
tempy = 1./((3*a+4*b)/3 - 4*(b-c)*(tempx/3));
plot(tempx,tempy/1e9,'or',LineWidth=2.0)
plot([tempx(1),tempx(1)],[0,tempy(1)/1e9], 'black--', LineWidth=1.5)  % 100
plot([tempx(2),tempx(2)],[0,tempy(2)/1e9], 'black--', LineWidth=1.5)  % 311
plot([tempx(3),tempx(3)],[0,tempy(3)/1e9], 'black--', LineWidth=1.5)  % 110, 112
plot([tempx(4),tempx(4)],[0,tempy(4)/1e9], 'black--', LineWidth=1.5)  % 111

grid on
xlim([-0.05,1.05])
ylim([160,285])
ax = gca;
ax.XTick = 0:0.25:1;
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = 0:0.125:1;
ax.YTick = 160:25:285;
ax.YAxis.MinorTick = 'on';
ax.YAxis.MinorTickValues = 160:5:285;

xlabel('$3\Gamma$', 'Interpreter', 'latex','FontSize',24);
ylabel('$E^{hkl}$ (GPa)', 'Interpreter', 'latex', 'FontSize', 24)
set(gca, "FontSize", 16, 'LineWidth',1.0, 'TickDir','out')
copygraphics(gca,'Resolution',300)

%% Constrained compliance tensor
function [a,b,c] = U(C11,C12,C44) 

    % Bulk modulus K and shear modulus G for a polycrystal
    K = (C11+2*C12)/3;
    G_roots = roots([8 5*C11+4*C12 -C44*(7*C11-4*C12) -C44*(C11-C12)*(C11+2*C12)]);
    G = G_roots(2);

    % Eshelby solution and constrained compliance tensor   
    gamma = K/(3*K+4*G);
    delta = 3*(K+2*G)/(5*(3*K+4*G));
    L0 = [3*K 2*G 2*G];
    M0 = L0.^-1;
    L1 = [C11+2*C12 C11-C12 2*C44];
    S  = [3*gamma 2*delta 2*delta];
    I  = [1 1 1];
    T  = (I + (S.*M0).*(L1-L0)).^(-1);
    U = T.*M0;
    a = U(1)/3;
    b = U(2)/2;
    c = U(3)/2;

end