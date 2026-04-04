#include "jacobi_kokkos.h"
#include <cmath>

std::vector<float> JacobiKokkos(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy) {

    const int n = static_cast<int>(b.size());
    if (n == 0) return {};

    Kokkos::View<float**> A("A", n, n);
    Kokkos::View<float*> B("B", n);
    Kokkos::View<float*> X("X", n);
    Kokkos::View<float*> X_new("X_new", n);

    auto A_host = Kokkos::create_mirror_view(A);
    auto B_host = Kokkos::create_mirror_view(B);

    for (int i = 0; i < n; i++) {
        B_host(i) = b[i];
        for (int j = 0; j < n; j++) {
            A_host(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(B, B_host);

    Kokkos::deep_copy(X, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS && !converged; iter++) {
        Kokkos::parallel_for("JacobiStep",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(int i) {

            float sum = 0.0f;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += A(i, j) * X(j);
                }
            }

            X_new(i) = (B(i) - sum) / A(i, i);
        });

        if (iter % 5 == 0) {
            float max_diff = 0.0f;

            Kokkos::parallel_reduce("Error",
                Kokkos::RangePolicy<>(0, n),
                KOKKOS_LAMBDA(int i, float& local_max) {
                float diff = Kokkos::fabs(X_new(i) - X(i));
                if (diff > local_max) local_max = diff;
            },
                Kokkos::Max<float>(max_diff)
            );

            if (max_diff < accuracy) {
                converged = true;
            }
        }

        Kokkos::deep_copy(X, X_new);
    }

    auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);

    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = X_host(i);
    }

    return result;
}