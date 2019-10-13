function prepare_test(N)
    A = sprand(N,N,0.1);
    A = A+A';
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
        %if (i < j)
        %    A(i,j) = 0;
        %end
    end
    end
        
    for i = 1:N
        %A(i,i)=1.;
        A(i,i)=0.5+rand();
    end
    vec = rand(N,1);
    
    write_mat('mat.dat',A);
    write_vec('rhs.dat',vec);    
end