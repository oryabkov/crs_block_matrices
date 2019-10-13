
#include <cstdlib>
#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include <utils/init_cuda.h>
#include <utils/log_mpi.h>
#include <utils/mpi_timer_event.h>
#include <utils/cuda_timer_event.h>
#include <for_each/for_each_1d.h>
#include <for_each/for_each_1d_cuda_impl.cuh>
#include <for_each/for_each_func_macro.h>
#include <communication/comm.h>
#include <communication/linear_partitioner.h>
#include <communication/parted_map.h>
#include <numerical_algos/vectors/block_vector.h>
#include <numerical_algos/vectors/block_vector_operations.h>
#include <numerical_algos/vectors/block_vector_operations_impl.h>
#include <numerical_algos/matrices/brs_matrix.h>
#include <numerical_algos/matrices/brs_matrix_impl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/cgs.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/jacobi.h>
#include <numerical_algos/lin_solvers/preconditioners/lu_sgs.h>

using namespace numerical_algos::lin_solvers;

typedef SCALAR_TYPE                                                     real;
typedef utils::log_mpi                                                  log_t;
typedef utils::cuda_timer_event                                         timer_event_t;
static const t_tensor_field_storage                                     storage = TFS_DEVICE;
static const t_for_each_type                                            for_each_type = FET_CUDA;
typedef for_each_1d<for_each_type>                                      for_each_t;
typedef communication::linear_partitioner                               partitioner_t;
typedef communication::parted_map<partitioner_t>                        map_t;
typedef numerical_algos::block_vector<real,storage,map_t>               vector_t;
typedef numerical_algos::block_vector_operations<real,storage,map_t>    vector_operations_t;
typedef numerical_algos::brs_matrix<real,storage,map_t>                 matrix_t;
typedef matrix_t::structure_type                                        matrix_structure_t;
typedef default_monitor<vector_operations_t,log_t>                      monitor_t;
typedef lu_sgs<matrix_t,vector_operations_t,log_t>                      prec_t;
typedef jacobi<matrix_t,prec_t,vector_operations_t,monitor_t,log_t>     lin_solver_jacobi_t;
typedef cgs<matrix_t,prec_t,vector_operations_t,monitor_t,log_t>        lin_solver_cgs_t;
typedef bicgstab<matrix_t,prec_t,vector_operations_t,monitor_t,log_t>   lin_solver_bicgstab_t;
typedef bicgstabl<matrix_t,prec_t,vector_operations_t,monitor_t,log_t>  lin_solver_bicgstabl_t;

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

void write_convergency(const std::string &fn, const std::vector<std::pair<int,real> > &conv, real tol)
{
    std::ofstream f(fn.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("write_convergency: error while opening file " + fn);

    for (int i = 0;i < conv.size();++i) {
        if (!(f << conv[i].first << " " << conv[i].second << " " << tol << std::endl)) 
            throw std::runtime_error("write_convergency: error while writing to file " + fn);
    }
}

/*struct vars_weights_t
{
    real w[BRS_MATRIX_MAX_BLOCKDIM];

    __DEVICE_TAG__ real       &operator[](int i) { return w[i]; }
    __DEVICE_TAG__ const real &operator[](int i)const { return w[i]; }
};*/

/*struct init_weights_func
{
    FOR_EACH_FUNC_PARAMS_HELP(init_weights_func, int, block_size, real*, w, vars_weights_t, vars_weights)

    __DEVICE_TAG__ void operator()(const int &i)const 
    {
        for (int j = 0;j < block_size;++j)
            w[i*block_size + j] = vars_weights[j];
    }
};*/

/*void init_scalar_prod_weights(log_t &log, map_t &map,int block_size, vector_t &w, const std::string &weights_fn, vector_operations_t &vec_ops)
{
    vars_weights_t          vars_weights;
    for_each_t              for_each;
    if (weights_fn == "none") {
        log.info_f("setting ident weights...");
        for (int j = 0;j < block_size;++j) vars_weights[j] = real(1.f);
    } else {
        log.info_f("reading weights from file %s...", weights_fn.c_str());
        std::ifstream    f(weights_fn.c_str(), std::ifstream::in);
        for (int j = 0;j < block_size;++j) 
            if (!(f >> vars_weights[j])) throw std::runtime_error("error while reading weights file " + weights_fn);
        f.close();
    }
    for_each( init_weights_func( block_size, w.ptr(), vars_weights ),
              map.min_own_loc_ind(), map.max_own_loc_ind()+1 );
    vec_ops.set_scalar_prod_weights(w);
}*/

int main(int argc, char **args)
{
    if (MPI_Init(&argc, &args) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Init call failed ; abort" << std::endl;
        return 1;
    }

    int     comm_rank = communication::get_comm_rank(),
            comm_size = communication::get_comm_size();
    log_t   log;

    if (argc < 13) {
        if (comm_rank == 0)
            std::cout << "USAGE: " << std::string(args[0]) << " <matrix_fn> <alpha> <beta> <rhs_fn> <max_iters> <rel_tol> <weights_fn> <lin_solver_type> <use_prec> <repeats_n> <result_fn> <convergency_fn>" << std::endl;
        if (MPI_Finalize() != MPI_SUCCESS) {
            std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
            return 2;
        }
        return 0;
    }
    std::string             mat_fn(args[1]), rhs_fn(args[4]), weights_fn(args[7]), 
                            res_fn(args[11]), conv_fn(args[12]);
    real                    alpha = atof(args[2]), 
                            beta = atof(args[3]);
    int                     max_iters = atoi(args[5]);
    real                    rel_tol = atof(args[6]);
    int                     lin_solver_type = atoi(args[8]);
    bool                    use_prec = atoi(args[9]);
    int                     repeats_n = atoi(args[10]);

    //TODO device number
    utils::init_cuda(1+comm_rank);
    log.info_f("initializing cusparse...");
    cusparseHandle_t        handle = 0;
    CUSPARSE_SAFE_CALL( cusparseCreate(&handle) );

    log.info_f("creating map and partitioner...");
    int                     glob_size = read_matrix_size(mat_fn);
    partitioner_t           partitioner(glob_size, comm_size, comm_rank);
    map_t                   map(partitioner, true);

    vector_t                res, rhs, w;
    matrix_structure_t      mat_str, diag_mat_str;
    matrix_t                mat, inv_diag_mat;

    log.info_f("pre init matrix structure from file %s...", mat_fn.c_str());
    mat_str.pre_init_from_file(&map, mat_fn);
    log.info_f("completing map...");
    map.complete();

    log.info_f("initializing matrix structure...");
    mat_str.init();
    mat_str.print_stat();
    log.info_f("initializing diagonal matrix structure...");
    diag_mat_str.init_diagonal(&map, mat_str.block_row_size());
    log.info_f("initializing matrix ...");
    mat.init(handle, &mat_str);
    log.info_f("initializing diagonal matrix ...");
    inv_diag_mat.init(handle, &diag_mat_str);
    log.info_f("reading matrix values from file %s...", mat_fn.c_str());
    mat.read_from_file(mat_fn);
    log.info_f("constructing matrix linear combination(%e*I + %e*A)...", alpha, beta);
    mat.add_mul_ident_matrix(alpha, beta);
    log.info_f("calculating matrix inversed diagonal...");
    mat.copy_inverted_diagonal(inv_diag_mat);
    log.info_f("multiplying matrix by its inversed diagonal...");
    mat.left_mul_by_diagonal_matrix(inv_diag_mat);

    //TODO regroup somehow
    vector_operations_t     vec_ops(&map, mat_str.block_row_size());
    log.info_f("reading RHS from file %s...", rhs_fn.c_str());
    vec_ops.start_use_vector(rhs);
    rhs.read_values_from_file(map, rhs_fn);
    log.info_f("initializing result vector...");
    vec_ops.start_use_vector(res);
    log.info_f("initializing scalar_prod weights...");
    vec_ops.start_use_vector(w);
    if (weights_fn != "none")
        w.read_values_from_file(map, weights_fn);
    else
        vec_ops.assign_scalar(real(1.f), w);
    vec_ops.set_scalar_prod_weights(w);

    //NOTE use res as temporal buffer
    log.info_f("multiplying RHS vector by matrix inversed diagonal...");
    vec_ops.assign(rhs, res);
    inv_diag_mat.apply(res, rhs);

    prec_t                  prec(&vec_ops, &log);
    lin_solver_jacobi_t     lin_solver_jacobi(&vec_ops, &log);
    lin_solver_cgs_t        lin_solver_cgs(&vec_ops, &log);
    lin_solver_bicgstab_t   lin_solver_bicgstab(&vec_ops, &log);
    lin_solver_bicgstabl_t  lin_solver_bicgstabl(&vec_ops, &log);
    monitor_t               *mon;

    if (lin_solver_type == 0)
        mon = &lin_solver_jacobi.monitor();
    else if (lin_solver_type == 1)
        mon = &lin_solver_cgs.monitor();
    else if (lin_solver_type == 2)
        mon = &lin_solver_bicgstab.monitor();
    else if (lin_solver_type == 3)
        mon = &lin_solver_bicgstabl.monitor();
    else
        throw std::runtime_error("unknown solvert type");

    //TODO get rid of matrix block diagonal

    log.info_f("multiplying RHS vector by matrix inversed diagonal...");
    mon->init(rel_tol, real(0.f), max_iters);
    mon->set_save_convergence_history(true);
    /*lin_solver_jacobi.monitor().init(rel_tol, real(0.f), max_iters);
    lin_solver_cgs.monitor().init(rel_tol, real(0.f), max_iters);
    lin_solver_bicgstab.monitor().init(rel_tol, real(0.f), max_iters);*/
    if (use_prec) {
        lin_solver_jacobi.set_preconditioner(&prec);
        lin_solver_cgs.set_preconditioner(&prec);
        lin_solver_bicgstab.set_preconditioner(&prec);
        lin_solver_bicgstabl.set_preconditioner(&prec);
    }

    lin_solver_bicgstab.set_use_precond_resid(true);
    lin_solver_bicgstab.set_resid_recalc_freq(10);
    lin_solver_bicgstabl.set_use_precond_resid(true);
    lin_solver_bicgstabl.set_basis_size(5);
    lin_solver_bicgstabl.set_resid_recalc_freq(10/5);

    log.info_f("solving...");

    timer_event_t       e1,e2;
    bool                res_flag;
    int                 iters_performed;
    e1.init(); e2.init();

    e1.record();
    for (int rep = 0;rep < repeats_n;++rep) {
        vec_ops.assign_scalar(real(0.f), res);

        bool    res_flag_;
        int     iters_performed_;

        if (lin_solver_type == 0) {
            res_flag_ = lin_solver_jacobi.solve(mat, rhs, res);
            //iters_performed = lin_solver_jacobi.monitor().iters_performed();
        } else if (lin_solver_type == 1) {
            res_flag_ = lin_solver_cgs.solve(mat, rhs, res);
            //iters_performed = lin_solver_cgs.monitor().iters_performed();
        } else if (lin_solver_type == 2) {
            res_flag_ = lin_solver_bicgstab.solve(mat, rhs, res);
            //iters_performed = lin_solver_bicgstab.monitor().iters_performed();
        } else if (lin_solver_type == 3) {
            res_flag_ = lin_solver_bicgstabl.solve(mat, rhs, res);
            //iters_performed = lin_solver_bicgstab.monitor().iters_performed();
        } else 
            throw std::runtime_error("unknown solvert type");
        iters_performed_ = mon->iters_performed();

        if (rep > 0) {
            if (res_flag != res_flag_) throw std::runtime_error("res_flag changed from try to try");
            if (iters_performed != iters_performed_) throw std::runtime_error("number of iterations changed from try to try");
        }
        res_flag = res_flag_;
        iters_performed = iters_performed_;
    }
    e2.record();

    double  time = e2.elapsed_time(e1)/repeats_n;

    log.info_f("time to solve: %e s", time/1000.);
    log.info_f("time per iteration: %e ms", time/iters_performed);

    if (res_flag) 
        log.info("lin_solver returned success result");
    else
        log.info("lin_solver returned fail result");

    if (res_fn != "none") res.write_to_file(map, res_fn);
    if (conv_fn != "none") write_convergency(conv_fn, mon->convergence_history(), mon->tol_out());

    e1.release(); e2.release();

    if (MPI_Finalize() != MPI_SUCCESS) {
        std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
        return 3;
    }

    return 0;
}