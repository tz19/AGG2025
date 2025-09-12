function full = symbolic2full(symbolic)
%% Function to expand the symbolic form into 4th order tensor
    d_ijkl = zeros(3,3,3,3);
    d_ijkl(1,1,1,1) = 1;
    d_ijkl(2,2,2,2) = 1;
    d_ijkl(3,3,3,3) = 1;

    I = eye(3);

    full1 = (symbolic(1) - symbolic(2)) * (1/3) * einsum('ij,kl->ijkl',I,I);
    full2 = symbolic(3) * (1/2) * (einsum('ik,jl->ijkl',I,I) + einsum('il,jk->ijkl',I,I));
    full3 = (symbolic(2) - symbolic(3)) * d_ijkl;

    full = full1 + full2 + full3;

end