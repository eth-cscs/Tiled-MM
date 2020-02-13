#pragma once

#include "util.hpp"
#include <memory>
#include <cassert>

namespace gpu {

template <typename T>
class pinned_buffer {
public:
    pinned_buffer() = default;
    pinned_buffer(int n_streams, tile_dim tile)
    : n_streams(n_streams)
    , tile(tile) {
        size_t buffer_size = n_streams * n_tiles_per_stream * tile.size();
        buffer_ = std::vector<T>(buffer_size);

        // pin the buffer
        auto status = gpu::runtime_api::host_register(
            buffer_.data(),
            buffer_size * sizeof(T),
            gpu::runtime_api::flag::HostRegisterDefault);
        gpu::check_runtime_status(status);

        current_tile_idx = std::vector<int>(n_streams);

        // std::vector<device_event> events(n_streams*n_tiles_per_stream);
        // buffer_events = std::move(events);
        buffer_events = std::vector<device_event>(n_streams * n_tiles_per_stream);
    }

    pinned_buffer& operator=(pinned_buffer&&) = default;

    ~pinned_buffer() {
        // unpin the buffer
        if (buffer_.size()) {
            auto status = gpu::runtime_api::host_unregister(buffer_.data());
            gpu::check_runtime_status(status);
        }
    }

    T* buffer(int stream_idx) {
        int tile_idx = current_tile_idx[stream_idx];
        assert(stream_idx >= 0 && stream_idx < n_streams);
        assert(tile_idx >= 0 && tile_idx < n_tiles_per_stream);

        int stream_offset = stream_idx * n_tiles_per_stream * tile.size();
        int tile_offset = tile_idx * tile.size();
        return &buffer_[stream_offset + tile_offset];
    }

    size_t size() {
        return buffer_.size();
    }

    void set_tile_dimensions(tile_dim new_tile_dim) {
        tile = new_tile_dim;
        size_t buffer_size = n_streams * n_tiles_per_stream * tile.size();
        buffer_ = std::vector<T>(buffer_size);
    }

    void set_num_streams(int num_streams) {
        n_streams = num_streams;
        size_t buffer_size = n_streams * n_tiles_per_stream * tile.size();
        // buffer_ = std::make_unique<T[]>(buffer_size_);
        buffer_ = std::vector<T>(buffer_size);
    }

    void set_streams_and_tiles(int num_streams, tile_dim new_tile) {
        n_streams = num_streams;
        tile = new_tile;
        size_t buffer_size = n_streams * n_tiles_per_stream * tile.size();
        buffer_ = std::vector<T>(buffer_size);
    }

    void advance_buffer(int stream_idx) {
            current_tile_idx[stream_idx] = (current_tile_idx[stream_idx] + 1) % n_tiles_per_stream;
    }

    int num_tiles_per_stream() {
        return n_tiles_per_stream;
    }

    void wait_buffer(int stream_idx) {
        int event_idx = stream_idx * n_tiles_per_stream + current_tile_idx[stream_idx];
        buffer_events[event_idx].wait();
    }

    void record_event(int stream_idx, device_event&& event) {
        int event_idx = stream_idx * n_tiles_per_stream + current_tile_idx[stream_idx];
        buffer_events[event_idx] = std::move(event);
    }

private:
    int n_tiles_per_stream = 2;
    int n_streams;
    tile_dim tile;
    // std::unique_ptr<T[]> buffer_;
    std::vector<T> buffer_;
    std::vector<int> current_tile_idx;

    std::vector<device_event> buffer_events;
};
}
