#ifndef __SCFD_RECT_MAP_H__
#define __SCFD_RECT_MAP_H__

#include <vector>
#include <stdexcept>
#include <sstream>
#include <utils/boost_property_tree_fwd.h>
#include <vecs_mats/t_vec_tml.h>
#include <vecs_mats/t_rect_tml.h>

namespace communication
{

template<class T,int Dim>
struct rect_map
{
    typedef     int                         Ord;

    typedef     t_vec_tml<T  ,Dim>          vec_t;
    typedef     t_vec_tml<Ord,Dim>          ord_vec_t;
    typedef     t_rect_tml<T  ,Dim>         rect_t;
    typedef     t_rect_tml<Ord,Dim>         ord_rect_t;

    Ord                     comm_size, my_rank;
    //domain_rect_glob and rect_loc contain stencil: [-stencil,szx+stencil]x...
    ord_rect_t              domain_rect_glob, rect_loc;
    ord_rect_t              own_rect_glob, own_rect_loc;
    ord_rect_t              own_rect_no_stencil_glob, own_rect_no_stencil_loc;
    //global coordinates of point with (0,0,0) local coordinates
    ord_vec_t               loc0_glob;
    //when inited proc_rects is of size comm_size
    //proc_rects[i] contains indexes rect owned by i-th process
    std::vector<ord_rect_t> proc_rects_glob;
    Ord                     stencil_len;
    Ord                     own_size, loc_size, total_size;
    vec_t                   h;

    template<class T2>
    std::string             vec2str_(const t_vec_tml<T2,Dim> &v)const
    {
        std::stringstream   ss;
        ss << "(";
        for (int j = 0;j < Dim;++j) {
            ss << v[j];
            if (j < Dim-1) ss << ",";
        }
        ss << ")";
        return ss.str();
    }

    ord_vec_t               ind2vec_(const ord_rect_t &r, Ord i)const
    {
        ord_vec_t   r_sz = r.calc_size();
        ord_vec_t   res(r.i1);
        Ord         i_ = i;
        for (int j = 0;j < Dim;++j) {
            res[j] += i_ % r_sz[j];
            i_ /= r_sz[j];
        }
        return res;
    }
    Ord                     vec2ind_(const ord_rect_t &r, const ord_vec_t &v)const
    {
        Ord     res(0);
        for (int j = Dim-1;j >= 0;--j) {
            if (j < Dim-1) res *= (r.i2[j] - r.i1[j]);
            res += v[j] - r.i1[j];
        }
        return res;   
    }
    ord_vec_t               loc2vec(Ord i_loc)const
    {
        return ind2vec_(rect_loc, i_loc);
    }
    ord_vec_t               glob2vec(Ord i_glob)const
    {
        return ind2vec_(domain_rect_glob, i_glob);
    }
    Ord                     vec2loc(const ord_vec_t &v_loc)const
    {
        return vec2ind_(rect_loc, v_loc);
    }
    Ord                     vec2glob(const ord_vec_t &v_glob)const
    {
        return vec2ind_(domain_rect_glob, v_glob);
    }
    ord_vec_t               loc2glob_vec(const ord_vec_t &v_loc)const
    {
        return loc0_glob + v_loc;
    }
    ord_vec_t               glob2loc_vec(const ord_vec_t &v_glob)const
    {
        return v_glob - loc0_glob;
    }
    Ord                     get_rank_vec(const ord_vec_t &v_glob)const 
    { 
        for (Ord rank = 0;rank < comm_size;++rank) {
            if (check_glob_owned_vec(v_glob, rank)) return rank;
        }
        throw std::logic_error("rect_map::get_rank_vec: vector does not belong to any proc");
        return -1;
    }
    bool                    check_glob_owned_vec(const ord_vec_t &v_glob, Ord rank)const
    {
        return proc_rects_glob[rank].is_own(v_glob);
    }
    bool                    check_glob_owned_vec(const ord_vec_t &v_glob)const
    {
        return check_glob_owned_vec(v_glob, my_rank);
    }

    rect_map() : comm_size(0), my_rank(0) {}

    void    init(int comm_size_, int my_rank_, int stencil_len_,
                 const boost::property_tree::ptree &cfg);
    
    //'read' before construction part:
    Ord     get_own_rank()const { return my_rank; }
    bool    check_glob_owned(Ord i)const { return check_glob_owned(i, get_own_rank()); }
    Ord     get_total_size()const { return total_size; }
    Ord     get_size()const { return own_size; }
    Ord     own_glob_ind(Ord i)const { return vec2glob(ind2vec_(proc_rects_glob[my_rank], i)); }

    //'construction' part:
    //rect_map just ignores this information
    void    add_stencil_element(Ord i) { }
    void    complete() { }

    //'read' after construction part:
    Ord     get_rank(Ord i)const { return get_rank_vec(glob2vec(i)); }
    bool    check_glob_owned(Ord i, Ord rank)const { return check_glob_owned_vec(glob2vec(i), rank); }
    Ord     loc2glob(Ord i_loc)const { return vec2glob(loc2glob_vec(loc2vec(i_loc))); }
    Ord     glob2loc(Ord i_glob)const { return vec2loc(glob2loc_vec(glob2vec(i_glob))); }
    Ord     own_loc_ind(Ord i)const { return vec2loc(ind2vec_(own_rect_loc, i)); }
    Ord     min_loc_ind()const { return 0; }
    Ord     max_loc_ind()const { return loc_size-1; }
    Ord     min_own_loc_ind()const { return 0; }
    Ord     max_own_loc_ind()const { return loc_size-1; }
    bool    check_glob_has_loc_ind(Ord i_glob)const { return rect_loc.is_own(glob2loc_vec(glob2vec(i_glob))); }
    bool    check_loc_has_loc_ind(Ord i_loc)const { return rect_loc.is_own(loc2vec(i_loc)); }
    bool    is_loc_glob_ind_order_preserv()const { return true; }
};

}

#endif
