#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sycl/sycl.hpp>
#define pai 3.14159265358979323846

using namespace sycl;

class Array
{
public:
    int height, width, step;
    int* array;

    Array(int h, int w, int s) : height(h), width(w), step(s)
    {
        array = new int[height * width];
    }
    ~Array()
    {
        delete[] array;
    }
};

struct Line
{
    float rho;
    float angle;
};

struct hough_cmp_gt
{
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const
    {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

void HoughLinesStandard(const Array& src, std::vector<Line>& lines, int type,
    float rho, float theta,
    int threshold, int& linesMax,
    double min_theta, double max_theta)
{
    int* image = src.array;
    float irho = 1 / rho;

    int step = src.step;
    int height = src.height;
    int width = src.width;

    int max_rho = width + height;
    int min_rho = -max_rho;

    int numangle = (int)((max_theta - min_theta) / theta) + 1;
    if (numangle > 1 && std::fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;

    int numrho = (int)(((max_rho - min_rho) + 1) / rho + 0.5);

    std::cout << "numangle:" << numangle << " numrho:" << numrho << std::endl;

    int* accum = new int[(numangle + 2) * (numrho + 2)]{ 0 };
    float* tabSin = new float[numangle];
    float* tabCos = new float[numangle];

    float ang = static_cast<float>(min_theta);
    for (int n = 0; n < numangle; ang += (float)theta, n++)
    {
        tabSin[n] = (float)(std::sin((double)ang) * irho);
        tabCos[n] = (float)(std::cos((double)ang) * irho);
    }

    // 创建SYCL队列
    sycl::queue q;
    // 创建输入数组的缓冲区
    sycl::buffer<int, 1> inputBuffer(image, sycl::range<1>(height * width));
    // 创建累加器数组的缓冲区
    sycl::buffer<int, 1> accumBuffer(sycl::range<1>((numangle + 2) * (numrho + 2)));
    // 创建结果数组的缓冲区
    sycl::buffer<int, 1> resultBuffer(sycl::range<1>(linesMax));

    // 填充累加器
    q.submit([&](sycl::handler& h) {
        auto input = inputBuffer.get_access<sycl::access::mode::read>(h);
        auto accum = accumBuffer.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(height * width), [=](sycl::id<1> idx) {
            int i = idx[0] / width;
            int j = idx[0] % width;

            if (input[idx] != 0) {
                for (int n = 0; n < numangle; n++) {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    sycl::atomic_ref<int>(accum[(n + 1) * (numrho + 2) + r + 1], h) += 1;
                }
            }
        });
    });

    // 寻找所有本地个最大值和排序
    q.submit([&](sycl::handler& h) {
        auto accum = accumBuffer.get_access<sycl::access::mode::read>(h);
        auto result = resultBuffer.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(numrho), [=](sycl::id<1> ridx) {
            int r = ridx[0];

            for (int n = 0; n < numangle; n++) {
                int base = (n + 1) * (numrho + 2) + r + 1;

                if (accum[base] > threshold &&
                    accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                    accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                {
                    sycl::atomic_ref<int>(result[0], h) += 1;
                }
            }
        });

        h.single_task([&]() {
            sycl::sort(result, result + linesMax, hough_cmp_gt(accum.get_pointer()));
        });
    });

    // 获取排序后的结果
    auto resultHost = resultBuffer.get_access<sycl::access::mode::read>();
    linesMax = std::min(linesMax, resultHost[0]);
    double scale = 1. / (numrho + 2);

    for (int i = 0; i < linesMax; i++)
    {
        Line line;
        int idx = resultHost[i];
        int n = (int)(idx * scale) - 1;
        int r = idx - (n + 1) * (numrho + 2) - 1;
        line.rho = (r - (numrho - 1) * 0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(line);
    }

    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
}

int main()
{
    ifstream file("edges.txt");
    if (!file.is_open()) {
        std::cout << "file_open_error!" << std::endl;
        return 0;
    }

    Array image(800, 800, 800);

    char pixel;
    int temp = 0;
    while (file >> pixel) {
        if (pixel != '\n') {
            image.array[temp++] = (int)pixel - 48;
        }
    }
    file.close();

    std::vector<Line> lines;
    int linesMax = INT_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    HoughLinesStandard(image, lines, 0, 1, pai / 180, 200, linesMax, 0, pai);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end - start;
    std::cout << "Total Time: " << total_duration.count() * 1000 << "ms" << std::endl;

    return 0;
}
