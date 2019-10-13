
#if !defined(TEST_CPU) && !defined(TEST_CUDA)
#define TEST_CPU
#endif

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <mpi.h>
#include <utils/device_tag.h>
#include <utils/system_timer_event.h>
#ifdef TEST_CUDA
#include <utils/cuda_timer_event.h>
#include <utils/init_cuda.h>
#endif
#include <vecs_mats/t_vec_tml.h>
#include <tensor_field/t_tensor_field_ndim_tml.h>
#include <for_each/for_each_ndim.h>
#ifdef TEST_CUDA
#include <for_each/for_each_1d_cuda_impl.cuh>
#include <for_each/for_each_ndim_cuda_impl.cuh>
#endif
#include <for_each/for_each_storage_types.h>
#include <communication/comm.h>
#include <communication/rect_map.h>
#include <communication/rect_map_init_impl.h>
#include <communication/mpi_distributor.h>
#include <communication/mpi_distributor_copy_kernels_impl.h>

static const int                                                dim = 3;
typedef float                                                   real;
typedef t_vec_tml<real,dim>                                     vec_t;
typedef t_vec_tml<int,dim>                                      ord_vec_t;
#ifdef TEST_CPU
static const t_for_each_ndim_type                               for_each_type = FET_SERIAL_CPU;
#endif
#ifdef TEST_CUDA
static const t_for_each_ndim_type                               for_each_type = FET_CUDA;
#endif
#ifdef TEST_OPENMP
static const t_for_each_ndim_type                               for_each_type = FET_OPENMP;
#endif
typedef for_each_ndim<for_each_type,dim>                        for_each_t;
static const t_tensor_field_storage                             storage_type = t_for_each_storage_type_helper<for_each_type>::storage;
typedef t_tensor0_field_ndim_tml<real,dim,storage_type>         tensor_field_t;
typedef communication::rect_map<real,dim>                       map_t;
typedef communication::mpi_distributor<storage_type,
                                       for_each_type>           distributor_t;
#ifdef TEST_CUDA
typedef utils::cuda_timer_event                                 timer_event_t;
#else
typedef utils::system_timer_event                               timer_event_t;
#endif

struct area_t
{
    int                 stencil_len;
    ord_vec_t           loc_sz, logic_i0;
    //int                 size;

    area_t(int stencil_len_, const ord_vec_t &own_sz_) : 
           stencil_len(stencil_len_), loc_sz(own_sz_)
    {
        //size = 1;
        for (int j = 0;j < dim;++j) {
            logic_i0[j] = -stencil_len;
            loc_sz[j] += 2*stencil_len;
            //size *= loc_sz[j];
        }
    }

    /*template<class T>
    bool                    alloc(T *&x)const
    {
        x = (T*)malloc(size*sizeof(T));
        if (x == NULL) return false;
        return true;
    }*/
    void                    alloc(tensor_field_t &tf)const
    {
        tf.init(loc_sz, logic_i0);
    }
    /*__DEVICE_TAG__  int     calc_index(const ord_vec_t &ind)const
    {
        return loc_sz[0]*(loc_sz[1]*(ind[2] + stencil_len) + ind[1] + stencil_len) + ind[0] + stencil_len;
    }*/
};

struct fill_scalar_func
{
    FOR_EACH_FUNC_PARAMS_HELP(fill_scalar_func, 
                              real, scalar, tensor_field_t, x)

    __DEVICE_TAG__ void operator()(const ord_vec_t &ind)const
    {
        x(ind) = scalar;
    }
};

/*void    fill_scalar(const area_t &a, real scalar, real *x)
{
    ord_vec_t   ind;
    for (ind[0] = -a.stencil_len;ind[0] < a.own_sz[0]+a.stencil_len;++ind[0])
    for (ind[1] = -a.stencil_len;ind[1] < a.own_sz[1]+a.stencil_len;++ind[1])
    for (ind[2] = -a.stencil_len;ind[2] < a.own_sz[2]+a.stencil_len;++ind[2]) {
        x[a.calc_index(ind)] = scalar;
    }
}*/

/*void    update_boundary(const area_t &a, const real *x_0)
{
    ord_vec_t   ind;
    for (ind[0] = 0;ind[0] < a.own_sz[0];++ind[0])
    for (ind[1] = 0;ind[1] < a.own_sz[1];++ind[1]) {
        ind[2] = -1;
        x_0[a.calc_index(ind)]
    }
}*/

struct poisson_iteration_func
{
    FOR_EACH_FUNC_PARAMS_HELP(poisson_iteration_func, 
                              vec_t, h, tensor_field_t, x_0, tensor_field_t, f, 
                              tensor_field_t, x_1)

    __DEVICE_TAG__ void operator()(ord_vec_t &ind)const
    {
        real        num = -f(ind),
                    den = real(0.f);
        for (int j = 0;j < dim;++j) {
            real    x_0_m, x_0_p;
            --ind[j];
            x_0_m = x_0(ind);
            ++ind[j];
            ++ind[j];
            x_0_p = x_0(ind);
            --ind[j];

            num += (x_0_m + x_0_p)/(h[j]*h[j]);
            den += real(2.f)/(h[j]*h[j]);
        }
        x_1(ind) = num/den;
    }
};

/*void    poisson_iteration(const area_t &a, const real *x_0, const real *f, 
                          real *x_1)
{
    ord_vec_t   ind;
    real        num, den;
    for (ind[0] = 0;ind[0] < a.own_sz[0];++ind[0])
    for (ind[1] = 0;ind[1] < a.own_sz[1];++ind[1])
    for (ind[2] = 0;ind[2] < a.own_sz[2];++ind[2]) {
        real        num = -f[a.calc_index(ind)],
                    den = real(0.f);
        for (int j = 0;j < dim;++j) {
            real    x_0_m, x_0_p;
            --ind[j];
            x_0_m = x_0[a.calc_index(ind)];
            ++ind[j];
            ++ind[j];
            x_0_p = x_0[a.calc_index(ind)];
            --ind[j];

            num += (x_0_m + x_0_p)/(a.h[j]*a.h[j]);
            den += real(2.f)/(a.h[j]*a.h[j]);
        }
        x_1[a.calc_index(ind)] = num/den;
    }
}*/

void write_out_pos_scalar_file(const char f_name[], const char v_name[], const map_t &map, const tensor_field_t &x)
{
    tensor_field_t::view_type   view;
    view.init(x, true);

    int ip, np;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &ip) != MPI_SUCCESS) throw std::runtime_error("write_out_file::MPI_Comm_rank failed");
    if (MPI_Comm_size(MPI_COMM_WORLD, &np) != MPI_SUCCESS) throw std::runtime_error("write_out_file::MPI_Comm_size failed");

    MPI_Barrier(MPI_COMM_WORLD);

    if (ip == 0) {
        FILE *stream;
        //stream = fopen( f_name, "a" );
        stream = fopen( f_name, "w" );

        fprintf( stream, "View");
        fprintf( stream, " '");
        fprintf( stream, "%s", v_name);
        fprintf( stream, "' {\n");
        fprintf( stream, "TIME{0};\n");
        fflush(stream);
        fclose(stream);
    }

    for (int curr_ip = 0;curr_ip < np;++curr_ip) {

    MPI_Barrier(MPI_COMM_WORLD);

    if (ip != curr_ip) continue;
    
    FILE *stream;
    stream = fopen( f_name, "a" );

    ord_vec_t   ind, ind_loc;
    for (ind[0] = map.own_rect_glob.i1[0];ind[0] < map.own_rect_glob.i2[0];++ind[0])
    for (ind[1] = map.own_rect_glob.i1[1];ind[1] < map.own_rect_glob.i2[1];++ind[1])
    for (ind[2] = map.own_rect_glob.i1[2];ind[2] < map.own_rect_glob.i2[2];++ind[2]) {

        ind_loc = map.glob2loc_vec(ind);

        real    par = view(ind_loc);
        //real    par = ind[2];

        fprintf( stream, "SH(");

        real    x_l = ind[0]*map.h[0], x_r = (ind[0]+1)*map.h[0],
                y_l = ind[1]*map.h[1], y_r = (ind[1]+1)*map.h[1];
        for (int j3 = 0;j3 < 2;++j3) {
            real    z = (ind[2]+j3)*map.h[2];
            fprintf( stream, "%f,%f,%f,", x_l, y_l, z );
            fprintf( stream, "%f,%f,%f,", x_r, y_l, z );
            fprintf( stream, "%f,%f,%f,", x_r, y_r, z );
            fprintf( stream, "%f,%f,%f",  x_l, y_r, z );
            if (j3 == 0) fprintf( stream, ",");
        }

        fprintf( stream, "){" );

        for (int j = 0;j < 8;++j) {
            fprintf( stream,"%e", par );
            if (j != 7) fprintf( stream, "," );
        }
        fprintf( stream, "};\n");
    }

    fflush(stream);
    fclose(stream);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (ip == 0) {
        FILE *stream;
        stream = fopen( f_name, "a" );
        fprintf( stream, "};\n");
        fflush(stream);
        fclose(stream);
    }

    view.release(false);
}

int main(int argc, char **args)
{
    if (MPI_Init(&argc, &args) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Init call failed ; abort" << std::endl;
        return 1;
    }

    int     comm_rank, comm_size;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Comm_rank call failed ; abort" << std::endl;
        return 2;
    }
    if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) {
        std::cout << "ERROR: MPI_Comm_size call failed ; abort" << std::endl;
        return 2;
    }

    if (argc < 2) {
        std::cout << "USAGE: " << args[0] << " <config_fn>" << std::endl;
        std::cout << "        <config_fn>: configuration file name" << std::endl;
        if (MPI_Finalize() != MPI_SUCCESS) {
            std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
            return 9;
        }
        return 0;
    }

    boost::property_tree::ptree     cfg;
    std::string                     pos_out_fn;
    int                             iters_num;
    try {
        std::string     config_fn(args[1]);
        read_info(config_fn, cfg);
        pos_out_fn = cfg.get<std::string>("pos_out_fn");
        iters_num = cfg.get<int>("iters_num");
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during config read: " << e.what() << " ; abort" << std::endl;
        return 3;
    }

#ifdef TEST_CUDA
    try {
        utils::init_cuda(cfg.get<int>( boost::str(boost::format("dev_number%d") % comm_rank) ));
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during cuda initialization: " << e.what() << " ; abort" << std::endl;
        return 3;
    }
#endif

    map_t                           map;
    distributor_t                   distributor;
    try {
        map.init(comm_size, comm_rank, 1, cfg);
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during distributor map init: " << e.what() << " ; abort" << std::endl;
        return 3;
    }

#ifdef TEST_CUDA
    distributor.for_each.block_size = 128;
#endif

    area_t                          area(map.stencil_len, map.own_rect_no_stencil_loc.i2);
    tensor_field_t                  x_0, x_1, f;
    try {
        std::cout << "allocating data..." << std::endl;
        area.alloc(x_0); area.alloc(x_1); area.alloc(f);
        std::cout << "done" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "ERROR: error while allocating fileds: " << e.what() << " ; abort" << std::endl;
        return 3;
    }
    for_each_t                      for_each;

#ifdef TEST_CUDA
    for_each.block_size = 128;
#endif

    try {
        std::cout << "initial filling..." << std::endl;
        for_each(fill_scalar_func(real(0.f), x_0), map.rect_loc);
        for_each(fill_scalar_func(real(0.f), x_1), map.rect_loc);
        for_each(fill_scalar_func(real(1.f), f  ), map.rect_loc);
        std::cout << "done" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during fill_zero(): " << e.what() << " ; abort" << std::endl;
        return 4;
    }

    try {
        std::cout << "pre init distributor..." << std::endl;
        ord_vec_t   ind;
        for (ind[0] = map.own_rect_no_stencil_glob.i1[0];ind[0] < map.own_rect_no_stencil_glob.i2[0];++ind[0])
        for (ind[1] = map.own_rect_no_stencil_glob.i1[1];ind[1] < map.own_rect_no_stencil_glob.i2[1];++ind[1])
        for (ind[2] = map.own_rect_no_stencil_glob.i1[2];ind[2] < map.own_rect_no_stencil_glob.i2[2];++ind[2]) {
            for (int j = 0;j < dim;++j) {
                for (int di = -map.stencil_len;di <= map.stencil_len;++di) {
                    ind[j] += di;
                    if (!map.check_glob_owned_vec(ind)) {
                        distributor.add_stencil_element(map, map.vec2glob(ind));
                    }
                    ind[j] -= di;
                }
            }
        }
        std::cout << "done" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during distributor pre init: " << e.what() << " ; abort" << std::endl;
        return 5;
    }

    try {
        std::cout << "init distributor..." << std::endl;
        map.complete();
        distributor.init(map);
        std::cout << "done" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during distributor init: " << e.what() << " ; abort" << std::endl;
        return 6;
    }

    try {
        std::cout << "iterating..." << std::endl;
        timer_event_t   timer_start, timer_end;
        timer_start.init(); timer_end.init();
        timer_start.record();
        for (int iter = 0;iter < iters_num;++iter) {
            //update_boundary(a, x_0);
            distributor.set_data(x_0.d, x_0.size(), 1);
            distributor.sync();
            //poisson_iteration(area, x_0, f, x_1);
            for_each(poisson_iteration_func(map.h, x_0, f, x_1), map.own_rect_no_stencil_loc);
            std::swap(x_0, x_1);
        }
        timer_end.record();
        std::cout << "total calculation time(s): " << timer_end.elapsed_time(timer_start)/1000. << std::endl;
        timer_start.release(); timer_end.release();
        std::cout << "done" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during iterations: " << e.what() << " ; abort" << std::endl;
        return 7;
    }

    try {
        if (pos_out_fn != "none") {
            std::cout << "writing output..." << std::endl;
            write_out_pos_scalar_file(pos_out_fn.c_str(), "x_res", map, x_0);
            std::cout << "done" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cout << "ERROR: error during result output: " << e.what() << " ; abort" << std::endl;
        return 8;
    }

    if (MPI_Finalize() != MPI_SUCCESS) {
        std::cout << "WARNING: MPI_Finalize call failed" << std::endl;
        return 9;
    }

    return 0;
}