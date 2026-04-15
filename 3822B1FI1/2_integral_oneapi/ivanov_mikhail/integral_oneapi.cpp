#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  if (start == end || count <= 0)
    return 0.0;

  if (start > end)
    std::swap(start, end);

  float result = 0.0f;
  float step = (end - start) / count;

  sycl::queue q(device);

  {
    sycl::buffer<float> result_buf(&result, 1);

    q.submit([&](sycl::handler& h) {
      auto reduction = sycl::reduction(result_buf, h, sycl::plus<float>());

      h.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> idx, auto& sum) {
        float x = start + step * (idx[0] + 0.5f);
        float y = start + step * (idx[1] + 0.5f);

        sum += sycl::sin(x) * sycl::cos(y);
      });
    });

    q.wait();
  }

  return result * step * step;
}