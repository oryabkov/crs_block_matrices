
To build test run:
make test_apply_single_gpu.bin

To test run:
./test_apply_single_gpu.bin mat.dat vec.dat res.dat 1 none 0
This will create file res.dat with result of matrix-vector multiplication
And then from Matlab run:
check_apply_test_matrix('mat.dat','vec.dat','res.dat')
This will out delta norm and files res_correct.dat and delta_res.dat
Or check versus file res_correct_ref.dat from repository.


