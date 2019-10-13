function check_apply_test_matrix(A_fn,vec_fn,res_fn)
    A = read_mat(A_fn);
    vec = read_vec(vec_fn);
    res = read_vec(res_fn);
    res_correct = A*vec;
    fprintf('delta_norm = %f\n', norm(res-res_correct,Inf));
    write_vec('res_correct.dat', res_correct);
    write_vec('delta_res.dat', res-res_correct);
end