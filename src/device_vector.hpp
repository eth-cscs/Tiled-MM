#pragma once
#include "util.hpp"

namespace gpu {

// ****************************************************************** 
// class definition
// ****************************************************************** 
template <typename T>
class device_vector {
public:
    device_vector() = default;

    device_vector(std::size_t n);

    // copy-constructor are not supported
    device_vector(device_vector& other) = delete;
    // copy-operator disabled
    device_vector& operator=(device_vector&) = delete;

    // assignment operators are supported
    device_vector& operator=(device_vector&& other);

    T* data();
    std::size_t size();

    void resize(int size);

    ~device_vector();

private:
    T* data_ = nullptr;
    std::size_t size_ = 0lu;
};

// ****************************************************************** 
// class implementation
// ****************************************************************** 
template <typename T>
device_vector<T>::device_vector(std::size_t n): 
    data_(malloc_device<T>(n)),
    size_(n) {}

// assignment operators are supported
template <typename T>
device_vector<T>& device_vector<T>::operator=(device_vector<T>&& other) {
    if (this != &other) {
        if (this->data_) {
            auto status = cudaFree(this->data_);
            cuda_check_status(status);
        }
        this->data_ = other.data_;
        other.data_ = nullptr;
        this->size_ = other.size_;
    }
    return *this;
}

template <typename T>
T* device_vector<T>::data() {
    return data_;
}

template <typename T>
std::size_t device_vector<T>::size() {
    return size_;
}

template <typename T>
device_vector<T>::~device_vector() {
    if (data_) {
        cudaFree(data_);
    }
}

template <typename T>
void device_vector<T>::resize(int size) {
    this->~device_vector();
    *this = device_vector(size);
}

}
