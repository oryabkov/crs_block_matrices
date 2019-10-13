
#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <communication/rect_map.h>
#include <communication/rect_map_init_impl.h>

int main()
{
    communication::rect_map<3>  map1, map2;
    boost::property_tree::ptree cfg;
    read_info("rect_map_test.cfg", cfg);
    int comm_size = 2, stencil_len = 5;

    map1.init(comm_size, 0, stencil_len, cfg);
    map2.init(comm_size, 1, stencil_len, cfg);

    std::cout << "map1.get_own_rank() = " << map1.get_own_rank() << std::endl;
    std::cout << "map2.get_own_rank() = " << map2.get_own_rank() << std::endl;

    std::cout << "map1.get_total_size() = " << map1.get_total_size() << std::endl;
    std::cout << "map2.get_total_size() = " << map2.get_total_size() << std::endl;

    return 0;
}