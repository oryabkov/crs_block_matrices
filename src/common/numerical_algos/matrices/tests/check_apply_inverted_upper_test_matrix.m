function check_apply_inverted_upper_test_matrix(A_fn,vec_fn,perm_fn,res_fn)
    A = read_mat(A_fn);
    vec = read_vec(vec_fn);
    res = read_vec(res_fn);
    
    fileID = fopen(perm_fn,'r');
    perm = fscanf(fileID,'%f',size(vec,1));
    fclose(fileID);
    
    perm_inv(perm) = 1:length(perm);
    perm_inv = perm_inv';
    
    %spy(A(perm,perm));
    %spy(A);
    
    U_perm = triu(A(perm,perm));
    vec_perm = vec(perm);
    res_correct_perm = U_perm\vec_perm;
    res_correct = res_correct_perm(perm_inv);
    
    fprintf('delta_norm = %f\n', norm(res-res_correct,Inf));
    write_vec('res_correct.dat', res_correct);
    write_vec('delta_res.dat', res-res_correct);
end