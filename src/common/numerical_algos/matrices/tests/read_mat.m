function [A] = read_mat(fn)
    fileID = fopen(fn,'r');
    header = fgets(fileID);
    header = fgets(fileID);
    sizes = fscanf(fileID,'%f',[1,3]);
    coords = fscanf(fileID,'%f',[3,sizes(3)]);
    A = sparse(sizes(1),sizes(2));
    for i = 1:sizes(3)
        A(coords(1,i)+1,coords(2,i)+1) = coords(3,i);
    end
    fclose(fileID);
end