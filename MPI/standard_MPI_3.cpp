#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#include <mpi.h>
#define pai 3.14159265358979323846

using namespace std;

//代表输入数组，由于可能是二维的，所有有高和宽
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

//代表直线
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




/*
* recvcounts 和 displs 在主进程中分配并使用，用于在 MPI_Gatherv 调用中指定各个进程的接收数量和偏移量。
* 在 MPI_Gather 中，非主进程的第二个参数和第三个参数应该是 NULL 和 0, 因为它们只发送数据而不接收。
* 创建了一个 global_accum 数组，在主进程中接收所有进程的累加器总和。
* 使用 MPI_Reduce 将每个进程的 accum 汇总到主进程的 global_accum。
* 在代码的最后，释放了所有动态分配的内存。
* 在 MPI_Gatherv 调用中，使用了 all_sort_buf.data() 来获取向量的底层数组指针。
*/

void HoughLinesStandard(const Array& src, vector<Line>& lines, int type,
                        float rho, float theta,
                        int threshold, int& linesMax,
                        double min_theta, double max_theta)
{
    // 计时开始
    auto start_total = chrono::high_resolution_clock::now();

    //基础参数预处理
    int* image = src.array;
    float irho = 1 / rho;

    int step = src.step;
    int height = src.height;
    int width = src.width;

    int max_rho = width + height;
    int min_rho = -max_rho;

    //需要遍历的角度数量
    int numangle = (int)((max_theta - min_theta) / theta) + 1;
    if (numangle > 1 && fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;
    //需要遍历的步数
    int numrho = (int)(((max_rho - min_rho) + 1) / rho + 0.5);

    //在每个进程中计算部分的累加器
    int* accum = new int[(numangle + 2) * (numrho + 2)]{0};
    //相应角度对应的三角函数值
    float* tabSin;
    float* tabCos;

    //检查进程数与进程编号
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int start_numangle,end_numangle;
    if(rank==size-1)
    {
        start_numangle=(numangle/size)*rank;
        end_numangle=numangle;
        //cout<<rank<<" start,end: "<<start_numangle<<","<<end_numangle<<endl;
    }
    else
    {
        start_numangle=(numangle/size)*rank;
        end_numangle=(numangle/size)*(rank+1);
        //cout<<rank<<" start,end: "<<start_numangle<<","<<end_numangle<<endl;
    }

    int i, j;
    std::vector<int> sort_buf;



    //正式步骤一：填充部分的累加器
    auto start_step1 = chrono::high_resolution_clock::now();
    tabSin = new float[end_numangle - start_numangle];
    tabCos = new float[end_numangle - start_numangle];
    float ang = static_cast<float>(min_theta) + start_numangle * theta;
    for (int n = 0; n < (end_numangle - start_numangle); ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
                for (int n = start_numangle; n < end_numangle; n++)
                {
                    int r = (int)(j * tabCos[n-start_numangle] + i * tabSin[n-start_numangle] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
        }

    auto end_step1 = chrono::high_resolution_clock::now();
    


    //正式步骤二：寻找所有本地个最大值
    auto start_step2 = chrono::high_resolution_clock::now();
    for (int r = 0; r < numrho; r++)
        for (int n = start_numangle; n < end_numangle; n++)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;

            if (accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                sort_buf.push_back(base);
        }
    auto end_step2 = chrono::high_resolution_clock::now();



    // 使用MPI的通信功能, 汇总所有进程的最大直线候选列表
    auto start_mpi = chrono::high_resolution_clock::now();

    vector<int> all_sort_buf;
    int local_size = sort_buf.size();
    int* local_sort_buf = new int[local_size];
    for (int i = 0; i < local_size; i++)
        local_sort_buf[i] = sort_buf[i];

    // 其他进程发送其最大直线候选列表的大小
    MPI_Gather(&local_size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

    int* recvcounts = new int[size];
    int* displs = new int[size];
    if (rank == 0)
    {
        // 主进程需要收集所有进程的局部大小来准备缓冲区
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int total_size = 0;
        for (int i = 0; i < size; i++)
        {
            displs[i] = total_size;
            total_size += recvcounts[i];
        }

        all_sort_buf.resize(total_size);
    }
    else
    {
        // 非主进程只需要发送它们的局部大小
        MPI_Gather(&local_size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // 使用Gatherv收集所有进程的sort_buf到主进程
    if (rank == 0)
        MPI_Gatherv(local_sort_buf, local_size, MPI_INT, all_sort_buf.data(), recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(local_sort_buf, local_size, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);

    auto end_mpi = chrono::high_resolution_clock::now();


    // 将每个进程的局部累加器发送到主进程中，并对位相加
    auto start_reduce = chrono::high_resolution_clock::now();
    int* global_accum = nullptr;
    if (rank == 0)
    {
        global_accum = new int[(numangle + 2) * (numrho + 2)]{0};
    }
    MPI_Reduce(accum, global_accum, (numangle + 2) * (numrho + 2), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    auto end_reduce = chrono::high_resolution_clock::now();



    //在主进程中进行最终的直线选择
    auto start_step3 = chrono::high_resolution_clock::now();
    if (rank == 0)
    {
        //正式步骤三：从大到小排序
        std::sort(all_sort_buf.begin(), all_sort_buf.end(), hough_cmp_gt(accum));

        //正式步骤四：存储前 min(total,linesMax) 条直线
        linesMax = std::min(linesMax, (int)all_sort_buf.size());
        double scale = 1. / (numrho + 2);

        for (i = 0; i < linesMax; i++)
        {
            Line line;
            int idx = all_sort_buf[i];
            int n = (int)(idx * scale) - 1;
            int r = idx - (n + 1) * (numrho + 2) - 1;
            line.rho = (r - (numrho - 1) * 0.5f) * rho;
            line.angle = static_cast<float>(min_theta) + n * theta;
            lines.push_back(line);
        }

        // ... (最终直线选择的代码)
        auto end_step3 = chrono::high_resolution_clock::now();

        // 计算并输出时间
        chrono::duration<double> duration_step1 = end_step1 - start_step1;
        chrono::duration<double> duration_step2 = end_step2 - start_step2;
        chrono::duration<double> duration_mpi = end_mpi - start_mpi;
        chrono::duration<double> duration_reduce = end_reduce - start_reduce;
        chrono::duration<double> duration_step3 = end_step3 - start_step3;
        chrono::duration<double> duration_total = end_step3 - start_total;

        cout << "Step 1 time: " << duration_step1.count() * 1000 << "ms" << endl;
        cout << "Step 2 time: " << duration_step2.count() * 1000 << "ms" << endl;
        cout << "MPI time: " << duration_mpi.count() * 1000 << "ms" << endl;
        cout << "Reduce time: " << duration_reduce.count() * 1000 << "ms" << endl;
        cout << "Step 3 time: " << duration_step3.count() * 1000 << "ms" << endl;
        cout << "Total time: " << duration_total.count() * 1000 << "ms" << endl;

        // 释放在主进程中额外分配的内存
        delete[] global_accum;
    }

    // 释放在所有进程中分配的内存
    delete[] local_sort_buf;
    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
    delete[] recvcounts;
    delete[] displs;
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
