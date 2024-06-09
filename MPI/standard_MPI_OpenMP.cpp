#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#include <mpi.h>
#include <utility>
#include <numeric>
#include <omp.h>
#define pai 3.14159265358979323846
#define ThreadNum 5

using namespace std;

// 代表输入数组，由于可能是二维的，所有有高和宽
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

// 代表直线
struct Line
{
    float rho;
    float angle;
};

bool cmp(pair<int, int> a, pair<int, int> b)
{
    return a.first > b.first; // 根据fisrt的值升序排序
}

void HoughLinesStandard(const Array& src, vector<Line>& lines, int type,
                        float rho, float theta,
                        int threshold, int& linesMax,
                        double min_theta, double max_theta)
{
    // 计时开始
    auto start_total = chrono::high_resolution_clock::now();
    
    // 基础参数预处理
    int* image = src.array;
    float irho = 1 / rho;

    int step = src.step;
    int height = src.height;
    int width = src.width;

    int max_rho = width + height;
    int min_rho = -max_rho;

    // 需要遍历的角度数量
    int numangle = (int)((max_theta - min_theta) / theta) + 1;
    if (numangle > 1 && fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;
    // 需要遍历的步数
    int numrho = (int)(((max_rho - min_rho) + 1) / rho + 0.5);

    // 在每个进程中计算部分的累加器
    int* accum;
    // 相应角度对应的三角函数值
    float* tabSin;
    float* tabCos;

    // 检查进程数与进程编号
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int start_numangle, end_numangle;
    if (rank == size - 1)
    {
        start_numangle = (numangle / size) * rank;
        end_numangle = numangle;
    }
    else
    {
        start_numangle = (numangle / size) * rank;
        end_numangle = (numangle / size) * (rank + 1);
    }

    int i, j;
    vector<int> local_values, local_indices;

    // 正式步骤一：填充部分的累加器
    auto start_step1 = chrono::high_resolution_clock::now();
    tabSin = new float[end_numangle - start_numangle];
    tabCos = new float[end_numangle - start_numangle];
    accum = new int[(end_numangle - start_numangle + 2) * (numrho + 2)]{0};
    float ang = static_cast<float>(min_theta) + start_numangle * theta;
    for (int n = 0; n < (end_numangle - start_numangle); ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }
    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++)
            {
                if (image[i * step + j] != 0)
                    for (int n = start_numangle; n < end_numangle; n++)
                    {
                        int r = (int)(j * tabCos[n - start_numangle] + i * tabSin[n - start_numangle] + 0.5);
                        r += (numrho - 1) / 2;
                        accum[(n - start_numangle + 1) * (numrho + 2) + r + 1]++;
                    }
            }
    }
    auto end_step1 = chrono::high_resolution_clock::now();

    // 正式步骤二：寻找所有本地个最大值
    auto start_step2 = chrono::high_resolution_clock::now();
    for (int r = 0; r < numrho; r++)
        for (int n = start_numangle; n < end_numangle; n++)
        {
            int base = (n - start_numangle + 1) * (numrho + 2) + r + 1;
            int position = (n + 1) * (numrho + 2) + r + 1;

            if (accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
            {
                local_values.push_back(accum[base]);
                local_indices.push_back(position);
            }
        }
    auto end_step2 = chrono::high_resolution_clock::now();

    // 使用MPI的通信功能, 汇总所有进程的最大直线候选列表
    auto start_mpi = chrono::high_resolution_clock::now();

    int local_size = local_values.size();
    int* recvcounts_values = new int[size];
    int* displs_values = new int[size];
    vector<int> all_values, all_indices;

    MPI_Gather(&local_size, 1, MPI_INT, recvcounts_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int total_size_values = std::accumulate(recvcounts_values, recvcounts_values + size, 0);
        all_values.resize(total_size_values);
        all_indices.resize(total_size_values);

        displs_values[0] = 0;
        for (int i = 1; i < size; i++)
        {
            displs_values[i] = displs_values[i - 1] + recvcounts_values[i - 1];
        }
    }

    MPI_Gatherv(local_values.data(), local_size, MPI_INT, all_values.data(), recvcounts_values, displs_values, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_indices.data(), local_size, MPI_INT, all_indices.data(), recvcounts_values, displs_values, MPI_INT, 0, MPI_COMM_WORLD);

    auto end_mpi = chrono::high_resolution_clock::now();

    // 在主进程中进行最终的直线选择
    auto start_step3 = chrono::high_resolution_clock::now();
    if (rank == 0)
    {
        // 重新组合数据
        vector<pair<int, int>> all_sort_buf;
        for (size_t i = 0; i < all_values.size(); ++i)
            all_sort_buf.push_back(make_pair(all_values[i], all_indices[i]));

        // 正式步骤三：从大到小排序
        std::sort(all_sort_buf.begin(), all_sort_buf.end(), cmp);

        // 正式步骤四：存储前 min(total,linesMax) 条直线
        linesMax = std::min(linesMax, (int)all_sort_buf.size());
        double scale = 1. / (numrho + 2);

        for (i = 0; i < linesMax; i++)
        {
            Line line;
            int idx = all_sort_buf[i].second;
            int n = (int)(idx * scale) - 1;
            int r = idx - (n + 1) * (numrho + 2) - 1;
            line.rho = (r - (numrho - 1) * 0.5f) * rho;
            line.angle = static_cast<float>(min_theta) + n * theta;
            lines.push_back(line);
        }

        auto end_step3 = chrono::high_resolution_clock::now();

        // 计算并输出时间
        chrono::duration<double> duration_step1 = end_step1 - start_step1;
        chrono::duration<double> duration_step2 = end_step2 - start_step2;
        chrono::duration<double> duration_mpi = end_mpi - start_mpi;
        chrono::duration<double> duration_step3 = end_step3 - start_step3;
        chrono::duration<double> duration_total = end_step3 - start_total;

        cout << "Step 1 time: " << duration_step1.count() * 1000 << "ms" << endl;
        cout << "Step 2 time: " << duration_step2.count() * 1000 << "ms" << endl;
        cout << "MPI time: " << duration_mpi.count() * 1000 << "ms" << endl;
        cout << "Step 3 time: " << duration_step3.count() * 1000 << "ms" << endl;
        cout << "Total time: " << duration_total.count() * 1000 << "ms" << endl;
    }

    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
    delete[] recvcounts_values;
    delete[] displs_values;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ifstream file("edges.txt");
    if (!file.is_open()) {
        cout << "file_open_error!" << endl;
        MPI_Finalize();
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

    vector<Line> lines;
    int linesMax = INT_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    HoughLinesStandard(image, lines, 0, 1, pai / 180, 200, linesMax, 0, pai);
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    if (rank == 0)
    {
        cout << "total_time: " << duration.count() * 1000 << "ms" << endl;
        cout << "linesMax: " << linesMax << endl;
        cout << "All the lines detected are below (rho+angle):\n";
        for (auto iter = lines.begin(); iter != lines.end(); iter++)
        {
            cout << iter->rho << " " << iter->angle << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
