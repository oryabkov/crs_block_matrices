function [v] = read_vec(fn)
    fileID = fopen(fn,'r');
    sizes = fscanf(fileID,'%f',[1,2]);
    %v = fscanf(fileID,'%f',[1,sizes(1)]);
    v = fscanf(fileID,'%f',sizes(1));
    fclose(fileID);
end