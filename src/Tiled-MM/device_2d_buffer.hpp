#pragma once
#include "device_2d_vector.hpp"
#include "tile_dim.hpp"
#include "util.hpp"

namespace gpu {

// ****************************************************************** 
// class definition
// ****************************************************************** 
template<typename T>
class device_2d_buffer {
public:
    device_2d_buffer() = default;
    device_2d_buffer(int streams, tile_dim tile)
            : n_streams(streams), tile_(tile) {
        for (int i = 0; i < streams; ++i) {
            d_vec.emplace_back(device_2d_vector<T>{tile.rows(), tile.cols(), 'C'});
        }
    }

    device_2d_buffer(device_2d_buffer&& other) = default;
    device_2d_buffer& operator=(device_2d_buffer&& other) = default;

    device_2d_buffer(const device_2d_buffer&) = delete;
    device_2d_buffer& operator=(const device_2d_buffer&) = delete;

    T* stream_buffer(int stream_id) {
        if (stream_id < 0 || stream_id >= n_streams) {
            std::runtime_error("stream id in device buffer has to be in the range [0, n_streams)");
        }
        return d_vec[stream_id].data();
    }

    void set_num_streams(int streams) {
        *this = device_2d_buffer(streams, tile_);
    }

    void set_tile_dimensions(tile_dim tile) {
        *this = device_2d_buffer(n_streams, tile);
    }

    tile_dim tile_dimensions() {
        return tile_;
    }

    void set_streams_and_tiles(int streams, tile_dim tile) {
        *this = device_2d_buffer(streams, tile);
    }

    std::size_t pitch() {
        if (d_vec.size()) {
            return d_vec[0].pitch;
        }
        return 0;
    }

private:
    int n_streams;
    tile_dim tile_;
    std::vector<device_2d_vector<T>> d_vec;
};
}

