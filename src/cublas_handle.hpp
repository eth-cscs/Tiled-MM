#pragma once

#include "util.hpp"

namespace gpu {
// wrapper around cublasHandle
class cublas_handle {
public:
    cublas_handle() {
        cudaSetDevice(0);
        auto status = 
        cublasCreate(&handle_);
        cublas_check_status(status);
        valid_ = true;
    }

    ~cublas_handle() {
        if (valid_) {
            // std::cout << "freeing cublas handle" << std::endl;
            auto status =
            cublasDestroy(handle_);
            cublas_check_status(status);
        }
    }

    // move constructor
    cublas_handle(cublas_handle&& other) {
        handle_ = other.handle_;
        valid_ = other.valid_;
        other.valid_ = false;
    }

    // move-assignment operator
    cublas_handle& operator=(cublas_handle&& other) {
        if (this != &other) {
            if (valid_) {
                auto status = 
                cublasDestroy(handle_);
                cublas_check_status(status);
            }
            handle_ = other.handle_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // copy-constructor disabled
    cublas_handle(cublas_handle&) = delete;
    // copy-operator disabled
    cublas_handle& operator=(cublas_handle&) = delete;

    // return the unerlying cublas handle
    cublasHandle_t handle() const {
        return handle_;
    }

private:
    bool valid_ = false;
    cublasHandle_t handle_;
};
}
