#pragma once
#include "device_vector.hpp"
#include "tile_dim.hpp"
#include "util.hpp"

namespace gpu {

// ****************************************************************** 
// class definition
// ****************************************************************** 
template<typename T>
class device_buffer {
public:
    device_buffer() = default;
    device_buffer(int streams, tile_dim tile);

    T* stream_buffer(int stream_id);

    T* data();

    void set_num_streams(int streams);

    void set_tile_dimensions(tile_dim tile);
    tile_dim tile_dimensions();

    void set_streams_and_tiles(int streams, tile_dim tile);

private:
    int n_streams;
    tile_dim tile_;
    device_vector<T> d_vec;
};

// ****************************************************************** 
// class implementation
// ****************************************************************** 
template<typename T>
device_buffer<T>::device_buffer(int streams, tile_dim tile): 
        n_streams(streams), tile_(tile) {
    d_vec = device_vector<T>(streams * tile_.size());
}

template<typename T>
T* device_buffer<T>::stream_buffer(int stream_id) {
    if (stream_id < 0 || stream_id >= n_streams) {
        std::runtime_error("stream id in device buffer has to be in the range [0, n_streams)");
    }
    int tile_size = tile_.size();
    int offset = stream_id * tile_size;
    return d_vec.data() + offset;
}

template<typename T>
T* device_buffer<T>::data() {
    return d_vec.data();
}

template<typename T>
void device_buffer<T>::set_num_streams(int streams) {
    n_streams = streams;
    d_vec.resize(n_streams * tile_.size());
}

template<typename T>
void device_buffer<T>::set_tile_dimensions(tile_dim tile) {
    tile_ = tile;
    d_vec.resize(n_streams * tile_.size());
}

template<typename T>
tile_dim device_buffer<T>::tile_dimensions() {
    return tile_;
}

template<typename T>
void device_buffer<T>::set_streams_and_tiles(int streams, tile_dim tile) {
    tile_ = tile;
    n_streams = streams;
    d_vec.resize(n_streams * tile_.size());
}
}
