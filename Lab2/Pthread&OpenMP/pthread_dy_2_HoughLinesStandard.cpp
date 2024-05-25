/********************************************************************************************************\
*                                               动态线程                                                  *
\********************************************************************************************************/
#include<iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#include <pthread.h>
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


int* image;
float irho;

int step;
int height;
int width;
int threshold_;

int max_rho;
int min_rho;

int numangle;
int numrho;

int* accum;
float* tabSin;
float* tabCos;

//设定线程数量
const int ThreadNum = 4;

std::vector<int> sort_buf;
std::vector<int> bufs[ThreadNum]; //分开放入vector之后再合并，减小等待时间

// 步骤一：填充累加器
void* step1(void* arg)
{
    int* t_id = (int *)arg;//对应线程的id

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
                for (int n = *t_id; n < numangle; n+=ThreadNum)
                {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
        }

    pthread_exit(NULL);
}

// 步骤二：寻找所有本地个最大值
void* step2(void* arg)
{
    int* t_id = (int *)arg;//对应线程的id

    for (int r = 0; r < numrho; r++)
        for (int n = *t_id; n < numangle; n+=ThreadNum)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;
        
            if (accum[base] > threshold_ &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                bufs[*t_id].push_back(base);
        }
    pthread_exit(NULL);
}


void
HoughLinesStandard(const Array &src, vector<Line> &lines, int type,
    float rho, float theta,
    int threshold, int &linesMax,
    double min_theta, double max_theta)
{
    //基础参数预处理
    image = src.array;
    irho = 1 / rho;

    step = src.step;
    height = src.height;
    width = src.width;
    threshold_ = threshold;

    max_rho = width + height;
    min_rho = -max_rho;

    //需要遍历的角度数量
    numangle = (int)((max_theta - min_theta) / theta) + 1;
    if (numangle > 1 && fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;
    //需要遍历的步数
    numrho = (int)(((max_rho - min_rho) + 1) / rho + 0.5);

    cout << "numangle:" << numangle << " numrho:" << numrho << endl;

    //accum是累加器
    accum = new int[(numangle + 2) * (numrho + 2)] {0};
    //相应角度对应的三角函数值
    tabSin = new float[numangle];
    tabCos = new float[numangle];

    float ang = static_cast<float>(min_theta);
    for (int n = 0; n < numangle; ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    int i, j;

    pthread_t step_1_thread[ThreadNum];
    pthread_t step_2_thread[ThreadNum];
    int t1_id[ThreadNum];
    int t2_id[ThreadNum];

    //正式步骤一：填充累加器
    auto start = std::chrono::high_resolution_clock::now();
    for (int t_id = 0; t_id < ThreadNum; t_id++)
    {
        t1_id[t_id]=t_id;
        pthread_create(&step_1_thread[t_id], NULL, step1, &t1_id[t_id]);
    }
    for (int t_id = 0; t_id < ThreadNum; t_id++) {
        pthread_join(step_1_thread[t_id], NULL);
    }
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "step1: " << duration.count() * 1000 << "ms" << endl;


    //正式步骤二：寻找所有本地个最大值
    start = std::chrono::high_resolution_clock::now();
    for (int t_id = 0; t_id < ThreadNum; t_id++)
    {
        t2_id[t_id]=t_id;
        pthread_create(&step_2_thread[t_id], NULL, step2, &t2_id[t_id]);
    }
    for (int t_id = 0; t_id < ThreadNum; t_id++) {
        pthread_join(step_2_thread[t_id], NULL);
    }
    for(int t=0; t<ThreadNum; t++)
    {
        sort_buf.insert(sort_buf.end(), bufs[t].begin(), bufs[t].end());
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "step2: " << duration.count() * 1000 << "ms" << endl;


    //正式步骤三：从大到小排序
    start = std::chrono::high_resolution_clock::now();
    std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "step3: " << duration.count() * 1000 << "ms" << endl;


    //正式步骤四：存储前 min(total,linesMax) 条直线
    start = std::chrono::high_resolution_clock::now();
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
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "step4: " << duration.count() * 1000 << "ms" << endl;

}

int main()
{
    ifstream file("edges.txt");
    if (!file.is_open()) {
        cout << "file_open_error!" << endl;
        return 0;
    }

    Array img(800, 800, 800);
    
    char pixel;
    int temp = 0;
    while (file >> pixel) {
        if (pixel != '\n') {
            img.array[temp++] = (int)pixel - 48;
        }
    }
    file.close();


    vector<Line> lines;
    int linesMax = INT_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    HoughLinesStandard(img, lines, 0, 1, pai / 180, 200, linesMax, 0, pai);
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "total_time: " << duration.count() * 1000 << "ms" << endl;
    
    cout << "linesMax: " << linesMax << endl;
    cout << "All the lines detected are below (rho+angle):\n";
    for (auto iter = lines.begin(); iter != lines.end(); iter++)
    {
        cout << iter->rho << " " << iter->angle << endl;
    }

    
    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;

    return 0;
}