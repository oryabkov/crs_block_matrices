function write_vec(fn,vec)
    fid = fopen(fn,'wt');
    fprintf(fid, '%d %d\n', size(vec,1), 1);
    fclose(fid);
    dlmwrite(fn,vec, '-append', 'delimiter', '\t', 'precision','%0.8e')
end