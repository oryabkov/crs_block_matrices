function check_test(A_fn,rhs_fn,res_fn)
    A = read_mat(A_fn);
    rhs = read_vec(rhs_fn);
    res = read_vec(res_fn);
    res_correct = A\rhs;
    fprintf('delta_norm = %f\n', norm(res-res_correct,Inf));
    write_vec('res_correct.dat', res_correct);
    write_vec('delta_res.dat', res-res_correct);
end