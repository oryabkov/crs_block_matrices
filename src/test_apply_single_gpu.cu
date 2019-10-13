
#include <string>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <utils/init_cuda.h>
#include <communication/serial_map.h>
#include <numerical_algos/vectors/block_vector.h>
#include <numerical_algos/matrices/brs_matrix.h>
#include <numerical_algos/matrices/brs_matrix_impl.h>

//TODO for_each default block size

typedef float                                               real;
static const t_tensor_field_storage                         storage = TFS_DEVICE;
typedef communication::serial_map                           map_t;
typedef numerical_algos::block_vector<real,storage,map_t>   vector_t;
typedef numerical_algos::brs_matrix<real,storage,map_t>     matrix_t;
typedef matrix_t::structure_type                            matrix_structure_t;

//TODO move it some common header and use in brs_matrix and brs_matrix_structure
int read_matrix_size(const std::string &fn)
{
    std::ifstream f(fn.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix_size: error while opening file " + fn);

    std::string     buf;
    int             algebraic_rows_n, algebraic_cols_n, 
                    algebraic_nonzeros_n;
    int             block_row_size_, block_col_size_;
    int             glob_rows_n_, glob_cols_n_;
    int             glob_nonzeros_n_;

    if (!getline(f,buf)) throw std::runtime_error("read_matrix_size: error while reading first line");
    if (!(f >> buf >> block_row_size_ >> block_col_size_)) throw std::runtime_error("read_matrix_size: error while read block sizes");
    if (block_row_size_ != block_col_size_) throw std::runtime_error("read_matrix_size: block is not square");
    if (!(f >> algebraic_rows_n >> algebraic_cols_n >> algebraic_nonzeros_n)) throw std::runtime_error("read_matrix_size: error while read sizes");
    if (algebraic_rows_n != algebraic_cols_n) throw std::runtime_error("read_matrix_size: matrix is not square");
    if (algebraic_rows_n%block_row_size_ != 0) throw std::runtime_error("read_matrix_size: matrix size is not divider of block size");
    if (algebraic_nonzeros_n%(block_row_size_*block_col_size_) != 0) throw std::runtime_error("read_matrix_size: matrix nonzero size is not divider of block size square");
    glob_rows_n_ = algebraic_rows_n/block_row_size_;
    glob_cols_n_ = algebraic_cols_n/block_col_size_;
    glob_nonzeros_n_ = algebraic_nonzeros_n/(block_row_size_*block_col_size_);

    return glob_rows_n_;
}

int main(int argc, char **args)
{
    if (argc < 7) {
        std::cout << "USAGE: " << std::string(args[0]) << " <matrix_fn> <vector_fn> <result_fn> <apply_type> <color_perm_fn> <dev_num>" << std::endl;
        std::cout << "    <matrix_fn> - name of matrix file in mm format with blocks extension" << std::endl;
        std::cout << "    <vector_fn> - name of vector to which operator should be applied file" << std::endl;
        std::cout << "    <result_fn> - name of file with result vector" << std::endl;
        std::cout << "    <apply_type> - type of applied operator: 1 - normal matrix apply, 2 - inverted lower, 3 - inverted upper" << std::endl;
        std::cout << "    <color_perm_fn> - file name where colored permutation will be written, use \"none\" to ommit" << std::endl;
        std::cout << "    <dev_num> - cuda device number" << std::endl;
        return 0;
    }
    std::string             mat_fn(args[1]), vec_fn(args[2]), res_fn(args[3]);
    int                     apply_type = atoi(args[4]);
    std::string             color_perm_fn(args[5]);
    int                     dev_num = atoi(args[6]);

    utils::init_cuda(dev_num);
    cusparseHandle_t        handle = 0;
    CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );

    int                     glob_size = read_matrix_size(mat_fn);
    map_t                   map(glob_size);

    vector_t                res, vec;
    matrix_structure_t      mat_str;
    matrix_t                mat;

    mat_str.pre_init_from_file(&map, mat_fn);
    map.complete();

    mat_str.init();
    mat_str.print_stat();
    mat.init(handle, &mat_str);
    mat.read_from_file(mat_fn);
    vec.init_from_file(map, vec_fn);
    res.init(map, vec.block_size());

    //vec.size()

    if (apply_type == 1)
        mat.apply(vec, res);
    else if (apply_type == 2)
        mat.apply_inverted_lower(vec, res);
    else if (apply_type == 3)
        mat.apply_inverted_upper(vec, res);
    else
        throw std::runtime_error("wrong apply_type argument");

    if (color_perm_fn != "none") mat_str.write_colored_perm(color_perm_fn);

    res.write_to_file(map, res_fn);

    return 0;
}