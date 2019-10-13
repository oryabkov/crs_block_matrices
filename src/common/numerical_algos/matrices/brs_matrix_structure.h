// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_BRS_MATRIX_STRUCTURE_H__
#define __SCFD_BRS_MATRIX_STRUCTURE_H__

//#define SCFD_BRS_MATRIX_ENABLE_MPI

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
#include <mpi.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/sequential_vertex_coloring.hpp>
#endif
#include <utils/cuda_safe_call.h>
#include <utils/device_tag.h>
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
#include <tensor_field/t_tensor_field_tml.h>
#include <communication/comm.h>
#include <communication/mpi_distributor.h>
#endif

//TODO only CUDA variant is supported for now
//TODO move definition from class body (too long, hard to read interface)

namespace numerical_algos
{

template<class T,t_tensor_field_storage storage,class Map>
class brs_matrix;

template<class T,t_tensor_field_storage storage,class Map>
class brs_matrix_structure
{
    friend class brs_matrix<T,storage,Map>;

    static const t_for_each_type                                    for_each_type = FET_CUDA;
    typedef Map                                                     map_t;
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
    typedef communication::mpi_distributor<storage, for_each_type>  distributor_t;
#endif
    //typedef brs_matrix<T,storage,Map>                               matrix_t;

    int     block_row_size_, block_col_size_;
    //all sizes below are 'block' sizes, 
    //i.e. real algebraic matrix size is 
    //block_row_size_*glob_rows_n_ x block_col_size_*glob_cols_n_
    //glob_ sizes are whole matrix sizes and loc_ are sizes for current process
    int     glob_rows_n_, glob_cols_n_;
    int     glob_nonzeros_n_;
    int     loc_rows_n_, loc_cols_n_;   
    int     loc_nonzeros_n_;

    bool    is_diagonal_;

    t_tensor0_field_tml<int,storage>    row_ptr_, col_ind_, colors_;
    //col_ind_no_borders_ is special variant of col_ind_, where all border 
    //elements has only diagonal cols; user for triangular inversion
    t_tensor0_field_tml<int,storage>    col_ind_no_borders_;
    int                                 border_colors_n_, colors_n_;
    //TODO temporal solution with vectors-members
    std::vector<int>                    crs_row_ptr, csr_col_ind;

    const map_t      *map_;
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
    distributor_t    distributor_;
    distributor_t    *colors_distributors_;
#endif

    //TODO don't use crs_row_ptr/csr_col_ind and glob_XX->loc_XX
    //we need another cpu buffer instead of crs_row_ptr/csr_col_ind
    int     find_elem_ptr(int glob_row, int glob_col)const
    {
        int loc_row = map_->glob2loc(glob_row);
        for (int ptr = crs_row_ptr[loc_row];ptr < crs_row_ptr[loc_row+1];++ptr) {
            if (csr_col_ind[ptr] == glob_col) return ptr;
        }
        std::cout << glob_row << " " << loc_row << " " << glob_col << std::endl;
        throw std::logic_error("brs_matrix_structure::find_elem_ptr: failed to find ptr");
    }

    brs_matrix_structure(const brs_matrix_structure &mat) { }
    brs_matrix_structure &operator=(const brs_matrix_structure &mat) { return *this; }
public:
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
    brs_matrix_structure() : colors_distributors_(NULL) {}    
#else
    brs_matrix_structure() {}
#endif

    int     block_row_size()const { return block_row_size_; }
    int     block_col_size()const { return block_col_size_; }
    //void    mesh_pre_init(Map *map, int block_row_size, int block_col_size)

    void    pre_init_from_file(Map *map, const std::string &fn)
    {
        map_ = map;

        std::ifstream f(fn.c_str(), std::ifstream::in);
        if (!f) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while opening file " + fn);

        //FILE    *f = fopen(fn.c_str(), "r");
        //if (f == NULL) throw std::runtime_error("read_mat: error while opening file");

        std::string     buf;
        int             algebraic_rows_n, algebraic_cols_n, 
                        algebraic_nonzeros_n;
        if (!getline(f,buf)) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while reading first line");
        //if (fgets(buf, 256, f) == NULL) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while reading first line");
        if (!(f >> buf >> block_row_size_ >> block_col_size_)) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while scanf");
        //if (fscanf(f,"%s %d %d", buf, &block_row_size_, &block_col_size_) != 3) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while scanf");
        if (block_row_size_ != block_col_size_) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: block is not square");
        if (!(f >> algebraic_rows_n >> algebraic_cols_n >> algebraic_nonzeros_n)) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while scanf");
        //if (fscanf(f,"%d %d %d", &N1, &N2, &NNZ) != 3) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while scanf");
        if (algebraic_rows_n != algebraic_cols_n) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: matrix is not square");
        if (algebraic_rows_n%block_row_size_ != 0) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: matrix size is not divider of block size");
        if (algebraic_nonzeros_n%(block_row_size_*block_col_size_) != 0) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: matrix nonzero size is not divider of block size square");
        glob_rows_n_ = algebraic_rows_n/block_row_size_;
        glob_cols_n_ = algebraic_cols_n/block_col_size_;
        glob_nonzeros_n_ = algebraic_nonzeros_n/(block_row_size_*block_col_size_);

        if (!map->is_loc_glob_ind_order_preserv()) throw std::logic_error("brs_matrix_structure::pre_init_from_file: map is not order preserving");

        //std::vector<int>    crs_row_ptr, csr_col_ind;
        crs_row_ptr.push_back(0);

        int curr_glob_row = 0;
        for (int i = 0;i < glob_nonzeros_n_;++i) {
            int     col, row;
            for (int ii1 = 0;ii1 < block_row_size_;++ii1)
            for (int ii2 = 0;ii2 < block_col_size_;++ii2) {
                int     col_, row_;
                T       val;
                if (!(f >> row_ >> col_ >> val)) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while reading vals");
                //if (fscanf(f,"%d %d %f", &row, &col, &val) != 3) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: error while scanf");
                row_ = row_/block_row_size_;
                col_ = col_/block_col_size_;
                if ((ii1 == 0)&&(ii2 == 0)) {
                    col = col_; row = row_;
                } else {
                    if ((col != col_)||(row != row_)) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: blocks are intermitted");
                }
            }

            //csrColInd[i] = col;
            while (row != curr_glob_row) {
                if (map->check_glob_owned(curr_glob_row)) {
                    crs_row_ptr.push_back(csr_col_ind.size());
                }
                curr_glob_row++;
            }
            if (map->check_glob_owned(row)) csr_col_ind.push_back(col);
        }
        /*while (map->get_size()+1 != crs_row_ptr.size()) {
            crs_row_ptr.push_back(csr_col_ind.size());
        }*/
        while (glob_rows_n_ != curr_glob_row) {
            if (map->check_glob_owned(curr_glob_row)) {
                crs_row_ptr.push_back(csr_col_ind.size());
            }
            curr_glob_row++;
        }
        if (map->get_size()+1 != crs_row_ptr.size()) throw std::logic_error("brs_matrix_structure::pre_init_from_file: final matrix size does not correspond to map size");

        //add elements to map stencil
        for (int i = 0;i < csr_col_ind.size();++i)
            map->add_stencil_element(csr_col_ind[i]);
    }
    //TODO realize add_to_map case
    template<class HostMesh>
    void    pre_init_from_mesh(const Map *map, const HostMesh &mesh, int block_row_size, int block_col_size, bool add_to_map = true)
    {
        map_ = map;
        block_row_size_ = block_row_size;
        block_col_size_ = block_col_size;
        glob_rows_n_ = glob_cols_n_ = map_->get_total_size();

        //??
        crs_row_ptr.push_back(0);
        for (int i_ = 0;i_ < map_->get_size();++i_) {
            int i = map_->own_glob_ind(i_);
            //ISSUE do we need to sort elements here? (these are global indices)
            #define PROCESS_INDEX(idx) if ((idx) != -1) {   \
                csr_col_ind.push_back((idx));               \
            }
//                map_->add_stencil_element((idx));           
            for (int j = 0;j < mesh.cv[i].faces_n;++j) {
                PROCESS_INDEX(mesh.cv[i].neighbours[j]);
            }
            PROCESS_INDEX(i);
            #undef PROCESS_INDEX
            crs_row_ptr.push_back(csr_col_ind.size());
        }
    }

    //called after map is complete
    void    init()
    {
        if (map_->min_loc_ind() != 0) throw std::logic_error("brs_matrix_structure::init: map local elements indexation does not start with zero - not supported case yet");

        //init local sizes
        loc_rows_n_ = map_->get_size();
        loc_cols_n_ = (map_->max_loc_ind() - map_->min_loc_ind())+1;
        loc_nonzeros_n_ = csr_col_ind.size();

        //init local matrix structure
        row_ptr_.init(loc_rows_n_+1); col_ind_.init(loc_nonzeros_n_);
        col_ind_no_borders_.init(loc_nonzeros_n_);
        t_tensor0_field_view_tml<int,storage>   row_ptr_view(row_ptr_, false), 
                                                col_ind_view(col_ind_, false);
        t_tensor0_field_view_tml<int,storage>   col_ind_no_borders_view(col_ind_no_borders_, false);
        for (int i = 0;i < loc_rows_n_+1;++i)
            row_ptr_view(i) = crs_row_ptr[i];
        /*for (int i = 0;i < loc_nonzeros_n_;++i)
            col_ind_view(i) = map_->glob2loc(csr_col_ind[i]);*/
        //sort elements by local index
        for (int i = 0;i < loc_rows_n_;++i) {
            std::vector<std::pair<int,int> >    tmp(crs_row_ptr[i+1]-crs_row_ptr[i]);
            for (int j = 0;j < crs_row_ptr[i+1]-crs_row_ptr[i];++j) {
                tmp[j].first  = map_->glob2loc(csr_col_ind[j + crs_row_ptr[i]]);
                tmp[j].second = csr_col_ind[j + crs_row_ptr[i]];
            }
            std::sort(tmp.begin(), tmp.end());
            for (int j = 0;j < crs_row_ptr[i+1]-crs_row_ptr[i];++j) {
                col_ind_view(j + crs_row_ptr[i]) = tmp[j].first;
                csr_col_ind[j + crs_row_ptr[i]] = tmp[j].second;
            }
            //std::sort(col_ind_view.d + crs_row_ptr[i], col_ind_view.d + crs_row_ptr[i+1]);
        }
        for (int i = 0;i < loc_nonzeros_n_;++i)
            col_ind_no_borders_view(i) = col_ind_view(i);
        row_ptr_view.release(true); col_ind_view.release(true);


#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        int comm_rank = communication::get_comm_rank(),
            comm_size = communication::get_comm_size();

        int                 border_elems_n = 0;
        std::vector<int>    packet_verts, packet_edges;
        int                 total_border_elems_n;
        std::vector<int>    packet_colors;

        for (int loc_row = 0;loc_row < map_->get_size();++loc_row) {
            bool is_border = false;
            for (int i = crs_row_ptr[loc_row];i < crs_row_ptr[loc_row+1];++i)
                if (!map_->check_glob_owned(csr_col_ind[i])) { is_border = true; break; }
            if (!is_border) continue;

            ++border_elems_n;

            packet_verts.push_back(map_->loc2glob(loc_row));
            packet_edges.push_back(crs_row_ptr[loc_row+1] - crs_row_ptr[loc_row]);
            for (int i = crs_row_ptr[loc_row];i < crs_row_ptr[loc_row+1];++i)
                packet_edges.push_back(csr_col_ind[i]);
        }

        if (comm_rank == 0) {
            //process 0 recieves number of boundary elements from each process
            std::vector<int>    proc_border_elems_n(comm_size);
            total_border_elems_n = border_elems_n;
            proc_border_elems_n[0] = border_elems_n;
            for (int rank = 1;rank < comm_size;++rank) {
                if (MPI_Recv(&(proc_border_elems_n[rank]), sizeof(proc_border_elems_n[rank]), MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::pre_init_from_file: MPI_Recv failed");
                total_border_elems_n += proc_border_elems_n[rank];
            }

            //process 0 recieves global indexes of boundary elements from each process
            //and forms new compact enumeration needed to build graph
            std::vector<int>    tmp_2_glob;
            std::map<int,int>   glob_2_tmp;
            for (int rank = 0;rank < comm_size;++rank) {
                std::vector<int>    packet_verts_rank;

                if (rank == 0) {
                    packet_verts_rank = packet_verts;
                } else {
                    packet_verts_rank.resize(proc_border_elems_n[rank]);
                    if (MPI_Recv(&(packet_verts_rank[0]), packet_verts_rank.size()*sizeof(int), MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
                }

                for (int i = 0;i < packet_verts_rank.size();++i) 
                    glob_2_tmp[packet_verts_rank[i]] = tmp_2_glob.size() + i;
                tmp_2_glob.insert(tmp_2_glob.end(), packet_verts_rank.begin(), packet_verts_rank.end());
            }

            //process 0 recieves rows of boundary elements from 
            //each process and build adjacency graph
            typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>     graph_t;
            typedef boost::property_map<graph_t, boost::vertex_index_t>::const_type         vertex_index_map_t;
            graph_t g(total_border_elems_n);

            int     tmp_i = 0;
            for (int rank = 0;rank < comm_size;++rank) {
                std::vector<int>    packet_edges_rank;

                if (rank == 0) {
                    packet_edges_rank = packet_edges;
                } else {
                    size_t  packet_edges_size;
                    if (MPI_Recv(&packet_edges_size,      sizeof(packet_edges_size),     MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
                    packet_edges_rank.resize(packet_edges_size);
                    if (MPI_Recv(&(packet_edges_rank[0]), packet_edges_size*sizeof(int), MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
                }

                for (int elem_i = 0, i = 0;elem_i < proc_border_elems_n[rank];++elem_i,++tmp_i) {
                    int edges_n = packet_edges_rank[i++];
                    for (int edge_i = 0;edge_i < edges_n;++edge_i) {
                        int adj_tmp_i = glob_2_tmp[packet_edges_rank[i++]];
                        add_edge(tmp_i, adj_tmp_i, g);
                    }
                }
            }

            //process 0 calculates colors
            std::vector<int>        border_elems_colors(total_border_elems_n);
            boost::iterator_property_map<int*, vertex_index_map_t> color(&border_elems_colors.front(), get(boost::vertex_index, g));
            border_colors_n_ = sequential_vertex_coloring(g, color);

            //zero color is reserver for internal elements
            colors_n_ = border_colors_n_+1;
            for (int i = 0;i < border_elems_colors.size();++i) ++border_elems_colors[i];

            //process 0 prepares packet with calculated colors
            packet_colors.resize(total_border_elems_n*2);
            for (int i = 0,packet_i = 0;i < border_elems_colors.size();++i) {
                packet_colors[packet_i++] = tmp_2_glob[i];
                packet_colors[packet_i++] = border_elems_colors[i];
            }

            //process 0 sends calculated colors to each process
            for (int rank = 1;rank < comm_size;++rank) {
                if (MPI_Send(&border_colors_n_ ,     sizeof(border_colors_n_),         MPI_BYTE, rank, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
                if (MPI_Send(&total_border_elems_n , sizeof(total_border_elems_n),     MPI_BYTE, rank, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
                if (MPI_Send(&(packet_colors[0])   , packet_colors.size()*sizeof(int), MPI_BYTE, rank, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
            }
        } else {
            //each process (except for 0) sends number of boundary elements to process 0
            if (MPI_Send(&border_elems_n      , sizeof(border_elems_n)          , MPI_BYTE, 0, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
            //each process (except for 0) sends global indexes of boundary elements to process 0
            if (MPI_Send(&(packet_verts[0])   , packet_verts.size()*sizeof(int) , MPI_BYTE, 0, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
            //each process (except for 0) sends rows of boundary elements to process 0
            size_t      packet_edges_size = packet_edges.size();
            if (MPI_Send(&packet_edges_size   , sizeof(packet_edges_size)       , MPI_BYTE, 0, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");
            if (MPI_Send(&(packet_edges[0])   , packet_edges.size()*sizeof(int) , MPI_BYTE, 0, 0, MPI_COMM_WORLD) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Send failed");

            //each process (except for 0) recieves number of colors from process 0
            if (MPI_Recv(&border_colors_n_    , sizeof(border_colors_n_)        , MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
            colors_n_ = border_colors_n_+1;
            //each process (except for 0) recieves total number of boundary elements from process 0
            if (MPI_Recv(&total_border_elems_n, sizeof(total_border_elems_n)    , MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
            //each process (except for 0) recieves elements colors from process 0
            packet_colors.resize(total_border_elems_n*2);
            if (MPI_Recv(&(packet_colors[0])  , packet_colors.size()*sizeof(int), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) throw std::runtime_error("brs_matrix_structure::init: MPI_Recv failed");
        }

#endif

        //each process inits colors_ array
        colors_.init(loc_cols_n_); 
        t_tensor0_field_view_tml<int,storage>   colors_view(colors_, false);
        //first, fill all with zero color (internal)
        for (int i = 0;i < loc_cols_n_;++i)
            colors_view(i) = 0;
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        //second, fill color of all border elements that are in process stencil
        for (int i = 0,packet_i = 0;i < total_border_elems_n;++i) {
            int     glob_i = packet_colors[packet_i++],
                    color = packet_colors[packet_i++];

            if (!map_->check_glob_has_loc_ind(glob_i)) continue;

            colors_view(map_->glob2loc(glob_i)) = color;
        }
#endif
        for (int i = 0;i < loc_rows_n_;++i) {
            if (colors_view(i) != 0) {
                for (int ptr = crs_row_ptr[i];ptr < crs_row_ptr[i+1];++ptr)
                    col_ind_no_borders_view(ptr) = i;
            }
        }

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        colors_distributors_ = new distributor_t[border_colors_n_];
        if (colors_distributors_ == NULL) throw std::runtime_error("brs_matrix_structure::init: failed to alloc colors_distributors_ array");
#endif

        for (int i = 0;i < loc_nonzeros_n_;++i) {
            int     glob_i = csr_col_ind[i];
            //pass own elemnts
            if (map_->check_glob_owned(glob_i)) continue;

            //init main distributor
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
            distributor_.add_stencil_element_pass_map(*map_, glob_i);
#endif
            //init color elements distributors 
            for (int color = 0;color < border_colors_n_;++color) {
                if (colors_view(map_->glob2loc(glob_i)) != color+1) continue;

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
                colors_distributors_[color].add_stencil_element_pass_map(*map_, glob_i);
#endif
            }
        }

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        distributor_.init(*map_);
        for (int color = 0;color < border_colors_n_;++color)
            colors_distributors_[color].init(*map_);
#endif

        colors_view.release(true); col_ind_no_borders_view.release(true);

        is_diagonal_ = false;
    }
    void    init_diagonal(const Map *map, int block_size)
    {
        map_ = map; block_row_size_ = block_size; block_col_size_ = block_size;

        glob_rows_n_ = glob_cols_n_ = glob_nonzeros_n_ = map_->get_total_size();
        loc_rows_n_ = loc_nonzeros_n_ = map_->get_size();
        //ISSUE it could be loc_cols_n_ = map_->get_size() as well, not sure
        loc_cols_n_ = (map_->max_loc_ind() - map_->min_loc_ind())+1;

        row_ptr_.init(loc_rows_n_+1); col_ind_.init(loc_nonzeros_n_);
        colors_.init(loc_cols_n_); 

        t_tensor0_field_view_tml<int,storage>   row_ptr_view(row_ptr_, false), 
                                                col_ind_view(col_ind_, false);
        t_tensor0_field_view_tml<int,storage>   colors_view(colors_, false);

        for (int i = 0;i < loc_rows_n_+1;++i) row_ptr_view(i) = i;
        for (int i = 0;i < loc_nonzeros_n_;++i) col_ind_view(i) = i;
        for (int i = 0;i < loc_cols_n_;++i) colors_view(i) = 0;

        row_ptr_view.release(true); col_ind_view.release(true);
        colors_view.release(true); 

        col_ind_no_borders_ = col_ind_;

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        distributor_.init(*map_);
#endif

        is_diagonal_ = true;
    }
    template<class HostMesh>
    void    init_from_mesh(const Map *map, const HostMesh &mesh, int block_row_size, int block_col_size)
    {
        pre_init_from_mesh(map, mesh, block_row_size, block_col_size, false);
        init();
    }
    void    print_stat()const
    {
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        int     ip = communication::get_comm_rank(),
                np = communication::get_comm_size();
#else
        int     ip = 0;
        int     np = 1;
#endif
        if (ip == 0) {
            std::cout << "brs_matrix_structure::print_stat: " << std::endl << std::flush;
        }
        for (int curr_ip = 0;curr_ip < np;++curr_ip) {

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if (ip != curr_ip) continue;

            std::cout << "process " << ip << " :" << std::endl;
            std::cout << "block_row_size = " << block_row_size_ << std::endl;
            std::cout << "block_col_size = " << block_col_size_ << std::endl;
            std::cout << "glob_rows_n = " << glob_rows_n_ << std::endl;
            std::cout << "glob_cols_n = " << glob_cols_n_ << std::endl;
            std::cout << "glob_nonzeros_n = " << glob_nonzeros_n_ << std::endl;
            std::cout << "loc_rows_n = " << loc_rows_n_ << std::endl;
            std::cout << "loc_cols_n = " << loc_cols_n_ << std::endl;
            std::cout << "loc_nonzeros_n = " << loc_nonzeros_n_ << std::endl;
            std::cout << std::flush;
        }
    }
    void    write_colored_perm(const std::string &fn)
    {
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        int     ip = communication::get_comm_rank(),
                np = communication::get_comm_size();
#else
        int     ip = 0;
        int     np = 1;
#endif
        if (ip == 0) {
            std::ofstream f(fn.c_str(), std::ofstream::out);
            if (!f) throw std::runtime_error("brs_matrix_structure::write_colored_perm: error while opening file " + fn);
            f.close();
        }
        t_tensor0_field_view_tml<int,storage>   colors_view(colors_, true);
        if (ip == 0) {
            std::cout << "output colors permutation to file " << fn << std::endl;
            std::cout << "colors number = " << colors_n_ << std::endl;
        }
        for (int curr_color = 0;curr_color < colors_n_;++curr_color) {
            for (int curr_ip = 0;curr_ip < np;++curr_ip) {

#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                if (ip != curr_ip) continue;

                std::ofstream f(fn.c_str(), std::ofstream::out | std::ofstream::app);
                if (!f) throw std::runtime_error("brs_matrix_structure::write_colored_perm: error while opening file " + fn);

                for(int i_ = 0;i_ < map_->get_size();i_++) {
                    int i_loc = map_->own_loc_ind(i_),
                        i_glob = map_->own_glob_ind(i_);
                    if (colors_view(i_loc) != curr_color) continue;
                    f << i_glob+1 << " ";
                }

                //This one is needed only for matrix visualization
                if (curr_color == 0) f << std::endl;

                f.close();
            }

            //This one is needed only for matrix visualization
            if (curr_color != 0) {
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                if (ip == 0) {
                    std::ofstream f(fn.c_str(), std::ofstream::out | std::ofstream::app);
                    if (!f) throw std::runtime_error("brs_matrix_structure::write_colored_perm: error while opening file " + fn);
                    f << std::endl;
                    f.close();
                }
            }

/*#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if (ip == 0) {
                std::ofstream f(fn.c_str(),  std::ofstream::out | std::ofstream::app);
                if (!f) throw std::runtime_error("brs_matrix_structure::write_colored_perm: error while opening file " + fn);
                f << " : ";
                f.close();
            }*/   

        }
        colors_view.release(false);
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    void    free()
    {
        if (row_ptr_.d != NULL) row_ptr_.free();
        if (col_ind_.d != NULL) col_ind_.free();
        if ((col_ind_no_borders_.d != NULL)&&(col_ind_no_borders_.own)) col_ind_no_borders_.free();
        if (colors_.d != NULL) colors_.free();
#ifdef SCFD_BRS_MATRIX_ENABLE_MPI
        if (colors_distributors_ != NULL) delete []colors_distributors_;
#endif
    }

    ~brs_matrix_structure()
    {
        free();
    }
};

}

#endif
