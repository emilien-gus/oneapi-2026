#include "block_gemm_oneapi.h"
#include <cassert>

std::vector<float> GemmBlockONEAPI(const std::vector<float> &a, const std::vector<float> &b, size_t size,
                                   sycl::device device)
{
    assert(a.size() == size * size);
    assert(b.size() == size * size);

    constexpr size_t TILE = 16;

    const size_t global_size = ((size + TILE - 1) / TILE) * TILE;

    std::vector<float> c(size * size, 0.0f);

    sycl::queue queue(device);

    {
        sycl::buffer<float> a_buffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float> b_buffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float> c_buffer(c.data(), sycl::range<1>(c.size()));

        queue.submit([&](sycl::handler &cgh) {
            auto matrix_a = a_buffer.get_access<sycl::access::mode::read>(cgh);
            auto matrix_b = b_buffer.get_access<sycl::access::mode::read>(cgh);
            auto matrix_c = c_buffer.get_access<sycl::access::mode::write>(cgh);

            sycl::local_accessor<float, 2> tile_a(sycl::range<2>(TILE, TILE), cgh);
            sycl::local_accessor<float, 2> tile_b(sycl::range<2>(TILE, TILE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_size, global_size), sycl::range<2>(TILE, TILE)),
                             [=](sycl::nd_item<2> item) {
                                 const size_t local_row = item.get_local_id(0);
                                 const size_t local_col = item.get_local_id(1);
                                 const size_t global_row = item.get_global_id(0);
                                 const size_t global_col = item.get_global_id(1);

                                 float value = 0.0f;
                                 const size_t block_count = (size + TILE - 1) / TILE;

                                 for (size_t block = 0; block < block_count; ++block)
                                 {
                                     const size_t a_col = block * TILE + local_col;
                                     const size_t b_row = block * TILE + local_row;

                                     if (global_row < size && a_col < size)
                                     {
                                         tile_a[local_row][local_col] = matrix_a[global_row * size + a_col];
                                     }
                                     else
                                     {
                                         tile_a[local_row][local_col] = 0.0f;
                                     }

                                     if (b_row < size && global_col < size)
                                     {
                                         tile_b[local_row][local_col] = matrix_b[b_row * size + global_col];
                                     }
                                     else
                                     {
                                         tile_b[local_row][local_col] = 0.0f;
                                     }

                                     item.barrier(sycl::access::fence_space::local_space);

                                     for (size_t k = 0; k < TILE; ++k)
                                     {
                                         value += tile_a[local_row][k] * tile_b[k][local_col];
                                     }

                                     item.barrier(sycl::access::fence_space::local_space);
                                 }

                                 if (global_row < size && global_col < size)
                                 {
                                     matrix_c[global_row * size + global_col] = value;
                                 }
                             });
        });

        queue.wait();
    }

    return c;
}
