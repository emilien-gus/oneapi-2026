#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(const std::vector<float>& A,
                                const std::vector<float>& b,
                                float accuracy) {
    
    int n = b.size();
    std::vector<float> x(n, 0.0f);
    
    Kokkos::View<float**> A_dev("A", n, n);
    Kokkos::View<float*> b_dev("b", n);
    Kokkos::View<float*> x_dev("x", n);
    Kokkos::View<float*> x_new_dev("x_new", n);
    Kokkos::View<float*> inv_diag("inv_diag", n);
    
    auto A_host = Kokkos::create_mirror_view(A_dev);
    auto b_host = Kokkos::create_mirror_view(b_dev);
    
    for (int i = 0; i < n; ++i) {
        b_host(i) = b[i];
        for (int j = 0; j < n; ++j) {
            A_host(i, j) = A[i * n + j];
        }
    }
    
    Kokkos::deep_copy(A_dev, A_host);
    Kokkos::deep_copy(b_dev, b_host);
    
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
        inv_diag(i) = 1.0 / A_dev(i, i);
        x_dev(i) = 0.0f;
    });
    
    int iteration = 0;
    float max_error = 0.0f;
    
    do {
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (j != i) sum += A_dev(i, j) * x_dev(j);
            }
            x_new_dev(i) = inv_diag(i) * (b_dev(i) - sum);
        });
        
        if (iteration % 8 == 0) {
            Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(int i, float& tmp) {
                float diff = Kokkos::fabs(x_new_dev(i) - x_dev(i));
                if (diff > tmp) tmp = diff;
            }, Kokkos::Max<float>(max_error));
        }
        
        Kokkos::deep_copy(x_dev, x_new_dev);
        iteration++;
        
    } while (iteration < ITERATIONS && max_error >= accuracy);
    
    auto x_result = Kokkos::create_mirror_view(x_dev);
    Kokkos::deep_copy(x_result, x_dev);
    for (int i = 0; i < n; ++i) x[i] = x_result(i);
    
    return x;
}