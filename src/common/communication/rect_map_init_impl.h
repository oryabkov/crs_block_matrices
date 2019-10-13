#ifndef __SCFD_RECT_MAP_INIT_IMPL_H__
#define __SCFD_RECT_MAP_INIT_IMPL_H__

#include <string>
#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/format.hpp>
#include <boost/math/special_functions/round.hpp>
#include "rect_map.h"

#define MAKE_FORMAT(STR,ARGS) boost::str(boost::format(STR)%ARGS)

namespace communication
{

template<class T, int Dim>
t_vec_tml<T,Dim> parse_vec_(const std::string &s)
{
    t_vec_tml<T,Dim>    res;
    std::stringstream   ss(s);

    for (int j = 0;j < Dim;++j) {
        if (!(ss >> res[j])) throw std::runtime_error("parse_vec_: failed to parse string " + s);
        if (j+1 < Dim) 
            if (!(ss.ignore(s.length(), ','))) throw std::runtime_error("parse_vec_: failed to parse string " + s);
    }

    //TODO check that there is nothing left in the string

    return res;
}

template<class T, int Dim>
t_rect_tml<T,Dim> init_rect_(const boost::property_tree::ptree &cfg)
{
    return t_rect_tml<T,Dim>(parse_vec_<T,Dim>(cfg.get<std::string>("p1")),
                             parse_vec_<T,Dim>(cfg.get<std::string>("p2")));
}

template<class T,int Dim>
void  rect_map<T,Dim>::init(int comm_size_, int my_rank_, int stencil_len_,
                            const boost::property_tree::ptree &cfg)
{
    comm_size = comm_size_;
    my_rank = my_rank_;
    stencil_len = stencil_len_;

    ord_vec_t             mesh_size = parse_vec_<Ord,Dim>(cfg.get<std::string>("mesh_size"));
    rect_t                domain_rect_coords = init_rect_<T,Dim>(cfg.get_child("domain_rect"));
    for (int j = 0;j < Dim;++j)
        h[j] = (domain_rect_coords.i2[j]-domain_rect_coords.i1[j])/mesh_size[j];

    domain_rect_glob = ord_rect_t(ord_vec_t::make_zero(), mesh_size);

    for (int j = 0;j < Dim;++j) {
        domain_rect_glob.i1[j] -= stencil_len;
        domain_rect_glob.i2[j] += stencil_len;
    }

    const boost::property_tree::ptree &part_cfg = cfg.get_child("partitioning").get_child(MAKE_FORMAT("np%d",comm_size));
    for (Ord rank = 0;rank < comm_size;++rank) {
        rect_t proc_rect_coords = init_rect_<T,Dim>(part_cfg.get_child(MAKE_FORMAT("proc%d_rect",rank)));
        ord_rect_t  proc_rect;
        for (int j = 0;j < Dim;++j) {
            proc_rect.i1[j] = boost::math::round(proc_rect_coords.i1[j]/h[j]);
            proc_rect.i2[j] = boost::math::round(proc_rect_coords.i2[j]/h[j]);
        }
        if (rank == my_rank) {
            own_rect_no_stencil_loc.i1 = ord_vec_t::make_zero();
            own_rect_no_stencil_loc.i2 = proc_rect.i2-proc_rect.i1;
            loc0_glob = proc_rect.i1;
            own_rect_no_stencil_glob = proc_rect;
            rect_loc = own_rect_no_stencil_loc;
        }
        for (int j = 0;j < Dim;++j) {
            if (proc_rect.i1[j] == 0) proc_rect.i1[j] -= stencil_len;
            if (proc_rect.i2[j] == mesh_size[j]) proc_rect.i2[j] += stencil_len;
        }
        proc_rects_glob.push_back(proc_rect);
    }

    own_rect_glob = proc_rects_glob[my_rank];
    own_rect_loc.i2 = own_rect_glob.i2-loc0_glob;
    own_rect_loc.i1 = own_rect_glob.i1-loc0_glob;

    for (int j = 0;j < Dim;++j) {
        rect_loc.i1[j] -= stencil_len;
        rect_loc.i2[j] += stencil_len;
    }

    own_size = own_rect_loc.calc_area();
    loc_size = rect_loc.calc_area();
    total_size = domain_rect_glob.calc_area();

    /*for (Ord rank = 0;rank < comm_size;++rank) {
        std::cout << rank << ": " << vec2str_(proc_rects_glob[rank].i1) << " " << vec2str_(proc_rects_glob[rank].i2) << std::endl;
    }*/
}

}

#endif
