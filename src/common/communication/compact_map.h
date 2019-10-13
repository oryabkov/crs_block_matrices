// Copyright © 2016 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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

#ifndef __SCFD_COMPACT_MAP_H__
#define __SCFD_COMPACT_MAP_H__

#include <cassert>
#include <vector>
#include <map>
#include <boost/foreach.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/range/algorithm/find.hpp>

namespace communication
{

//supposed to satisfy MAP concept

struct compact_map
{
    int                     total_size;
    std::vector<int>        starts, ends;
    int                     my_rank;
    std::map<int,int>       stencil_glob_2_loc;
    std::vector<int>        l_stencil_loc_2_glob;
    std::vector<int>        r_stencil_loc_2_glob;

    //creates uninitialized map
    compact_map() {}
    //dummy constructor
    compact_map(int N, int        comm_size, int _my_rank)
    {
        //if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) throw std::runtime_error("t_simple_map::init::MPI_Comm_size failed");
        int     sz_per_proc = (int)ceil(float(N)/comm_size);
        int     curr = 0;
        for (int iproc = 0;iproc < comm_size;++iproc) {
            starts.push_back(curr);
            curr += sz_per_proc; if (curr > N) curr = N;
            ends.push_back(curr);
        }
        total_size = N;
        my_rank = _my_rank;
    }

    //'read' before construction part:
    int     get_own_rank()const { return my_rank; }
    bool    check_glob_owned(int i)const { return check_glob_owned(i, get_own_rank()); }
    int     get_total_size()const { return total_size; }
    //WARNING //get_size(int rank) is not part of MAP concept anymore
    int     get_size(int rank)const { return ends[rank]-starts[rank]; }
    int     get_size()const { return get_size(get_own_rank()); }
    //in this map 'i' in own_glob_ind and own_loc_ind simply coincides with local index (not true for general MAP)
    int     own_glob_ind(int i)const
    {
        return i + starts[get_own_rank()];
    }

    //'construction' part:
    //t_simple_map just ignores this information
    //i is glob index here
    void    add_stencil_element(int i)
    {
        if (check_glob_owned(i)) return;
        //before complete() stencil_glob_2_loc simply store all stencil elements global indices
        stencil_glob_2_loc[i] = 0;
    }
    void    complete()
    {
        typedef std::pair<const int,int> pair_int_int;
        BOOST_FOREACH(pair_int_int &e, stencil_glob_2_loc) {
            if (e.first < starts[get_own_rank()]) {
                //to the left
                l_stencil_loc_2_glob.push_back(e.first);
            } else if (e.first >= ends[get_own_rank()]) {
                //to the right
                r_stencil_loc_2_glob.push_back(e.first);
            } else {
                //it's an logical error
                assert(0);
            }
        }
        boost::sort(l_stencil_loc_2_glob);
        boost::sort(r_stencil_loc_2_glob);
        for (int i = 0;i < l_stencil_loc_2_glob.size();++i) {
            int     glob_ind = l_stencil_loc_2_glob[i],
                loc_ind = -l_stencil_loc_2_glob.size() + i;
            stencil_glob_2_loc[glob_ind] = loc_ind;
        }
        for (int i = 0;i < r_stencil_loc_2_glob.size();++i) {
            int     glob_ind = r_stencil_loc_2_glob[i],
                loc_ind = get_size() + i;
            stencil_glob_2_loc[glob_ind] = loc_ind;
        }
    }

    //'read' after construction part:
    int     get_rank(int i)const
    {
        int     res = 0;
        while (i >= ends[res]) ++res;
        return res;
    }
    bool    check_glob_owned(int i, int rank)const { return ((i >= starts[rank])&&(i < ends[rank])); }
    int     loc2glob(int i_loc)const
    {
        if (i_loc < 0) {
            return l_stencil_loc_2_glob[i_loc + l_stencil_loc_2_glob.size()];
        } else if (i_loc >= get_size()) {
            return r_stencil_loc_2_glob[i_loc - get_size()];
        } else {
            return i_loc + starts[get_own_rank()];
        }
    }
    int     glob2loc(int i_glob)const
    {
        if (check_glob_owned(i_glob))
            return i_glob - starts[get_own_rank()];
        else
            return stencil_glob_2_loc.at(i_glob);
            //return stencil_glob_2_loc[i_glob];
    }
    int     own_loc_ind(int i)const
    {
        return i;
    }
    int     min_loc_ind()const
    {
        return -l_stencil_loc_2_glob.size();
    }
    int     max_loc_ind()const
    {
        return r_stencil_loc_2_glob.size()-1 +  get_size();
    }
    int     min_own_loc_ind()const
    {
        return 0;
    }
    int     max_own_loc_ind()const
    {
        return get_size()-1;
    }
    bool    check_glob_has_loc_ind(int i_glob)const
    {
        if (check_glob_owned(i_glob)) return true;
        return stencil_glob_2_loc.find(i_glob) != stencil_glob_2_loc.end();
    }
    bool    check_loc_has_loc_ind(int i_loc)const
    {
        return (i_loc >= min_loc_ind())&&(i_loc <= max_loc_ind());
    }
    bool  is_loc_glob_ind_order_preserv()const
    {
        return true;
    }
};

}

#endif
