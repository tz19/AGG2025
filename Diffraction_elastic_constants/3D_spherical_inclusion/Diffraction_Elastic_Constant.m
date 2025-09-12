function E = diffractionElasticConst3D(n,C11,C12,C44)
    [a,b,c] = U(C11,C12,C44);
    h = n(1);
    k = n(2);
    l = n(3);
    gamma = (h^2*k^2+l^2*k^2+h^2*l^2)/(h^2+k^2+l^2)^2;
    S = (3*a+4*b)/3 - 4*(b-c)*gamma;
    E = 1/S;
end

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