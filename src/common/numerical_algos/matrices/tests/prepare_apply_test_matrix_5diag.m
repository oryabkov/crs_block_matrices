function prepare_apply_test_matrix(N)
    B = ones(N,5);
    d = [-10 -1 0 1 10];
    A = spdiags(B,d,N,N);
    [row col v] = find(A);
    for i = 1:nnz(A)
        A(row(i),col(i)) = rand();
    end
    
    for i = 1:N
    for j = 1:N
        %if ((i <= N/2)&&(j > N/2))
        %    A(i,j) = 0;
        %end
        %if ((i > N/2)&&(j <= N/2))
        %    A(i,j) = 0;
        %end
    end
    end
        
    for i = 1:N
        A(i,i)=1.;
    end
    vec = rand(N,1);
    
    write_mat('mat.dat',A);
    %dlmwrite('mat.dat',[row-1 col-1 v], '-append', 'delimiter', '\t', 'precision','%0.8e');
    
    write_vec('vec.dat',vec);    
end