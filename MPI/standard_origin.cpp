#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#define pai 3.14159265358979323846

using namespace std;


//代表输入数组，由于可能是二维的，所有有高和宽
class Array
{
public:
    int height, width, step;
    int* array;

    Array(int h, int w, int s):height(h),width(w),step(s)
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

void
HoughLinesStandard(const Array &src, vector<Line> &lines, int type,
    float rho, float theta,
    int threshold, int &linesMax,
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

    cout << "numangle:" << numangle << " numrho:" << numrho << endl;

    //accum是累加器
    int* accum = new int[(numangle + 2) * (numrho + 2)] {0};
    //相应角度对应的三角函数值
    float* tabSin = new float[numangle];
    float* tabCos = new float[numangle];

    int i, j;
    std::vector<int> sort_buf;

    //正式步骤一：填充累加器
    auto start_step1 = chrono::high_resolution_clock::now();
    float ang = static_cast<float>(min_theta);
    for (int n = 0; n < numangle; ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
                for (int n = 0; n < numangle; n++)
                {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
        }
    auto end_step1 = chrono::high_resolution_clock::now();
    // 计算时间差

    //正式步骤二：寻找所有本地个最大值
    auto start_step2 = chrono::high_resolution_clock::now();
    for (int r = 0; r < numrho; r++)
        for (int n = 0; n < numangle; n++)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;
        
            if (accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                sort_buf.push_back(base);
        }
    auto end_step2 = chrono::high_resolution_clock::now();

    //正式步骤三：从大到小排序
    auto start_step3 = chrono::high_resolution_clock::now();
    std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));

    //正式步骤四：存储前 min(total,linesMax) 条直线
    linesMax = std::min(linesMax, (int)sort_buf.size());
    double scale = 1. / (numrho + 2);

    for (i = 0; i < linesMax; i++)
    {
        Line line;
        int idx = sort_buf[i];
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
    chrono::duration<double> duration_step3 = end_step3 - start_step3;
    chrono::duration<double> duration_total = end_step3 - start_total;

    cout << "Step 1 time: " << duration_step1.count() * 1000 << "ms" << endl;
    cout << "Step 2 time: " << duration_step2.count() * 1000 << "ms" << endl;
    cout << "Step 3 time: " << duration_step3.count() * 1000 << "ms" << endl;
    cout << "Total time: " << duration_total.count() * 1000 << "ms" << endl;

    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
}

int main()
{
    ifstream file("edges.txt");
    if (!file.is_open()) {
        cout << "file_open_error!" << endl;
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
    cout << "total_time: " << duration.count() * 1000 << "ms" << endl;
    
    
    cout << "linesMax: " << linesMax << endl;
    cout << "All the lines detected are below (rho+angle):\n";
    for (auto iter = lines.begin(); iter != lines.end(); iter++)
    {
        cout << iter->rho << " " << iter->angle << endl;
    }
    

    return 0;
}