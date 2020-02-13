#pragma once
#include "util.hpp"
#include "gpu_runtime_api.hpp"
#include <string>

namespace gpu {
template <typename T>
class device_2d_vector {
public:
    device_2d_vector() = default;

    device_2d_vector(int n_rows, int n_cols, char ordering) {
        ordering = std::toupper(ordering);
        // malloc_pitch assumes row major
        this->n_rows = ordering == 'R' ? n_rows : n_cols;
        this->n_cols = ordering == 'R' ? n_cols : n_rows;
        this->ordering = ordering;
        size_ = n_rows * n_cols;

        data_ = malloc_pitch_device<T>(this->n_rows, this->n_cols, &pitch);
    }
    // copy-constructor are not supported
    device_2d_vector(device_2d_vector& other) = delete;
    // copy-operator disabled
    device_2d_vector& operator=(device_2d_vector&) = delete;

    device_2d_vector(device_2d_vector&& other) {
        *this = std::forward<device_2d_vector>(other);
    }

    // move-assignment operators are supported
    device_2d_vector& operator=(device_2d_vector&& other) {
        if (this != &other) {
            if (this->data_) {
                auto status = runtime_api::free(this->data_);
                check_runtime_status(status);
            }
            this->data_ = other.data_;
            other.data_ = nullptr;
            this->size_ = other.size_;
            this->pitch = other.pitch;
            this->n_rows = other.n_rows;
            this->n_cols = other.n_cols;
            this->ordering = other.ordering;
        }
        return *this;
    }

    T* data() {
        return data_;
    }

    std::size_t size() {
        return size_;
    }

    void resize(int n_rows, int n_cols, char ordering) {
        this->~device_2d_vector();
        *this = device_2d_vector(n_rows, n_cols, ordering);
    }

    ~device_2d_vector() {
        if (data_) {
            runtime_api::free(data_);
        }
    }

    std::size_t pitch = 0u;
private:
    T* data_ = nullptr;
    std::size_t size_ = 0lu;
    std::size_t n_rows = 0u;
    std::size_t n_cols = 0u;
    char ordering;
};
}
