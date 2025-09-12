function E = diffractionElasticConst2D(n,Q,C11,C12,C44)

    eta1 = (C11 + 2*C12)/3;
    eta2 = (C11 - C12)/2;
    eta3 = C44;
    K_bar = (C11+2*C12)/3;
    G_roots = roots([8 5*C11+4*C12 -C44*(7*C11-4*C12) -C44*(C11-C12)*(C11+2*C12)]);
    mu_bar = G_roots(2);
    nu = (3*K_bar - 2*mu_bar)/(2*(3*K_bar + mu_bar));

    L_bar = symbolic2full([3*K_bar, 2*mu_bar, 2*mu_bar]);
    M_bar = symbolic2full([3*K_bar, 2*mu_bar, 2*mu_bar].^-1);
    L0 = symbolic2full([3*eta1, 2*eta2, 2*eta3]);
    L = einsum('ip,jq,kr,ls,pqrs->ijkl',Q,Q,Q,Q,L0);
    S = EshelbyCylinder(nu);
    I = symbolic2full([1,1,1]);

    U = Utensor_full(I,L,L_bar,S,M_bar);

    S = 0;
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    S = S + U(i,j,k,l) * n(i) * n(j) * n(k) * n(l);
                end
            end
        end
    end
    
    E = 1/S;

end