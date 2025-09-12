function L_inv = tensorInv4(L)

    Lv  = reshape(L, [9, 9]);
    Lv_inv = pinv(Lv);
    L_inv = reshape(Lv_inv,[3,3,3,3]);

    temp = einsum('ijkl,klmn->ijmn',L,L_inv);
    I = symbolic2full([1 1 1]);
    error = abs(temp - I);
    if any(error > 1e-12) 
        disp('wrong')
        stop
    end

end