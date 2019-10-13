function visualize_permuted_matrix(mat_fn, perm_fn)
A = read_mat(mat_fn);
spy(A);
szA = size(A,1);
fileID = fopen(perm_fn,'r');
perm = [];
blocks = [];
while ~feof(fileID)
    str = fgets(fileID);
    perm_ = sscanf(str,'%f');
    %sz_perm = size(perm_,1);
    blocks = [blocks; size(perm_,1)];
    perm = [perm; perm_];
end
fclose(fileID);
A_perm = A(perm,perm);
fig=figure; 
hax=axes; 
hold on 
spy(A_perm)
curr_x = 0.5
for i = 1:(size(blocks,1)+1)
    line([curr_x curr_x], [0.5 szA+0.5],'Color',[1 0 0]);
    line([0.5 szA+0.5], [curr_x curr_x],'Color',[1 0 0]);
    if i <= size(blocks,1) 
        curr_x = curr_x + blocks(i);
    end
end