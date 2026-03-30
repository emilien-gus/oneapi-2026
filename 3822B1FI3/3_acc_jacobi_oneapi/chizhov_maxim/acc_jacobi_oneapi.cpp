#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return std::vector<float>();
    }

    std::vector<float> x(n, 0.0f);

    sycl::queue queue(device);

    sycl::buffer<float> a_buf(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<float> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<float> x_new_buf((sycl::range<1>(n)));

    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < ITERATIONS) {
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                size_t row = i[0];
                float sum = 0.0f;
                float diag = a_acc[row * n + row];

                for (size_t j = 0; j < n; j++) {
                    if (j != row) {
                        sum += a_acc[row * n + j] * x_acc[j];
                    }
                }

                x_new_acc[row] = (b_acc[row] - sum) / diag;
                });
            });

        queue.wait();

        converged = true;

        {
            auto x_acc = x_buf.get_host_access();
            auto x_new_acc = x_new_buf.get_host_access();

            for (size_t i = 0; i < n; i++) {
                float diff = std::fabs(x_new_acc[i] - x_acc[i]);

                if (diff >= accuracy) {
                    converged = false;
                }

                x_acc[i] = x_new_acc[i];
            }
        }

        iteration++;
    }

    {
        auto x_acc = x_buf.get_host_access();
        for (size_t i = 0; i < n; i++) {
            x[i] = x_acc[i];
        }
    }

    return x;
}
