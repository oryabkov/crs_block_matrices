function write_mat(fn,A)
    %we need this transpose and exchange of col and row
    %because we need row sorted elements, not col sorted
    [col row v] = find(A');
    fid = fopen(fn,'wt');
    fprintf(fid, '%%MatrixMarket matrix coordinate real general\n');
    fprintf(fid, '%%AMGX 1 1\n');
    fprintf(fid, '%d %d %d\n', size(A,1), size(A,2), nnz(A));
    for i = 1:nnz(A)
        fprintf(fid, '%d %d %0.8e\n', row(i)-1, col(i)-1, v(i));
    end
    fclose(fid);
end