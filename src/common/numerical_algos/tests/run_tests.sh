#!/bin/bash
for np in 1 2 3 4
do
echo Run $np gpu result in 'res_'$np'gpu.dat'
/usr/local/ompi_3_1_2/bin/mpiexec -np $np ./test_apply mat.dat vec.dat 'res_'$np'gpu.dat' 1 'perm_'$np'gpu.dat'
done