#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>

#define pai 3.14159265358979323846

// CUDA核函数，在GPU上并行计算Hough变换
__global__ void HoughLinesCUDA(int* image, int width, int height, int* accum, float* tabSin, float* tabCos, int numangle, int numrho, int threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float irho = 1.0f / numrho;

    for (int i = idx; i < height * width; i += stride)
    {
        int x = i % width;
        int y = i / width;

        if (image[i] != 0)
        {
            for (int n = 0; n < numangle; n++)
            {
                int r = (int)(x * tabCos[n] + y * tabSin[n] + 0.5);
                r += (numrho - 1) / 2;
                atomicAdd(&accum[(n + 1) * (numrho + 2) + r + 1], 1);
            }
        }
    }
}

void HoughLinesParallel(const Array& src, std::vector<Line>& lines, int type, float rho, float theta, int threshold, int& linesMax, double min_theta, double max_theta)
{
    int* image = src.array;
    float irho = 1 / rho;

    int step = src.step;
    int height = src.height;
    int width = src.width;

    int max_rho = width + height;
    int min_rho = -max_rho;

    int numangle = (int)((max_theta - min_theta) / theta) + 1;
    if (numangle > 1 && fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;

    int numrho = (int)(((max_rho - min_rho) + 1) / rho + 0.5);

    std::cout << "numangle:" << numangle << " numrho:" << numrho << std::endl;

    int* accum = new int[(numangle + 2) * (numrho + 2)]{ 0 };
    float* tabSin = new float[numangle];
    float* tabCos = new float[numangle];

    float ang = static_cast<float>(min_theta);
    for (int n = 0; n < numangle; ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    // 创建并分配CUDA内存
    int* d_image;
    float* d_tabSin;
    float* d_tabCos;
    int* d_accum;
    cudaMalloc((void**)&d_image, sizeof(int) * height * width);
    cudaMalloc((void**)&d_tabSin, sizeof(float) * numangle);
    cudaMalloc((void**)&d_tabCos, sizeof(float) * numangle);
    cudaMalloc((void**)&d_accum, sizeof(int) * (numangle + 2) * (numrho + 2));

    // 将数据从主机内存复制到GPU内存
    cudaMemcpy(d_image, image, sizeof(int) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabCos, tabCos, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accum, accum, sizeof(int) * (numangle + 2) * (numrho + 2), cudaMemcpyHostToDevice);

    // 设置CUDA核函数的块和线程的数量
    int blockSize = 256;
    int numBlocks = (height * width + blockSize - 1) / blockSize;

    // 调用CUDA核函数进行并行计算
    HoughLinesCUDA<<<numBlocks, blockSize>>>(d_image, width, height, d_accum, d_tabSin, d_tabCos, numangle, numrho, threshold);

    // 将结果从GPU内存复制回主机内存
    cudaMemcpy(accum, d_accum, sizeof(int) * (numangle + 2) * (numrho + 2), cudaMemcpyDeviceToHost);

    // 寻找所有本地个最大值和排序
    std::vector<int> result;
    for (int r = 0; r < numrho; r++)
    {
        for (int n = 0; n < numangle; n++)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;

            if (accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
            {
                result.push_back(base);
            }
        }
    }

    // 对结果进行排序
    std::sort(result.begin(), result.end(), [&](int l1, int l2) {
        return accum[l1] > accum[l2] || (accum[l1] == accum[l2] && l1 < l2);
    });

    linesMax = std::min(linesMax, static_cast<int>(result.size()));
    double scale = 1. / (numrho + 2);

    for (int i = 0; i < linesMax; i++)
    {
        Line line;
        int idx = result[i];
        int n = (int)(idx * scale) - 1;
        int r = idx - (n + 1) * (numrho + 2) - 1;
        line.rho = (r - (numrho - 1) * 0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(line);
    }

    // 释放CUDA内存
    cudaFree(d_image);
    cudaFree(d_tabSin);
    cudaFree(d_tabCos);
    cudaFree(d_accum);

    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
}

int main()
{
    // 读取图像数据到Array对象中
    std::ifstream file("edges.txt");
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
    HoughLinesParallel(image, lines, 0, 1, pai / 180, 200, linesMax, 0, pai);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end - start;
    std::cout << "Total Time: " << total_duration.count() * 1000 << "ms" << std::endl;

    return 0;
}
