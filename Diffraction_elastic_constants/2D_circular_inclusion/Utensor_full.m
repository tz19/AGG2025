function U = Utensor_full(I,L1,L0,S,M0)

    temp1 = L1 - L0;
    temp2 = einsum('ijkl,klmn->ijmn',M0,temp1);
    temp3 = einsum('ijkl,klmn->ijmn',S,temp2);
    temp4 = temp3 + I;
    T = tensorInv4(temp4);
    U = einsum('ijkl,klmn->ijmn',T,M0);

end