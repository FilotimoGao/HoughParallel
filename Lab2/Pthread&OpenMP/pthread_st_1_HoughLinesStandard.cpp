/********************************************************************************************************\
*                                               静态线程                                                  *
\********************************************************************************************************/
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#include <semaphore.h>
#include <pthread.h>
#define pai 3.14159265358979323846

using namespace std;

//设定线程数量
const int ThreadNum = 4;

//所有可以挪出去的变量
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

//accum是累加器
int* accum;
//相应角度对应的三角函数值
float* tabSin;
float* tabCos;

//用于排序的
std::vector<int> sort_buf;

// 在全局范围定义互斥锁
pthread_mutex_t sort_buf_mutex;

//信号量定义
sem_t sem_leader;
sem_t sem_Division[ThreadNum-1];
sem_t sem_Elimination[ThreadNum-1];

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


struct MyThreadParams {
    Array *src;
    vector<Line> *lines;
    int type;
    float rho;
    float theta;
    int threshold;
    int linesMax;
    double min_theta;
    double max_theta;
    int t_id;
    MyThreadParams(){}
};

void*
HoughLinesStandard(void* arg)
{
    // 将参数转换为正确的类型
    MyThreadParams* params = static_cast<MyThreadParams*>(arg);

    // 提取参数
    const Array *src = params->src;
    vector<Line> *lines = params->lines;
    float rho = params->rho;
    float theta = params->theta;
    int threshold = params->threshold;
    int linesMax = params->linesMax;
    double min_theta = params->min_theta;
    double max_theta = params->max_theta;
    int t_id = params->t_id;
    
    //由于三角函数是共用的，分开会浪费空间，所以直接放到函数之外定义，由thread0提前计算完成
    if(t_id==0)
    {
        //基础参数预处理
        image = src->array;
        irho = 1 / rho;

        step = src->step;
        height = src->height;
        width = src->width;
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

        accum = new int[(numangle + 2) * (numrho + 2)] {0};
        tabSin = new float[numangle];
        tabCos = new float[numangle];

        float ang = static_cast<float>(min_theta);
        for (int n = 0; n < numangle; ang += (float)theta, n++)
        {
            tabSin[n] = (float)(sin((double)ang) * irho);
            tabCos[n] = (float)(cos((double)ang) * irho);
        }
        for(int t = 0; t < ThreadNum-1; t++)
        {
            sem_post(&sem_Division[t]);
        }
    }
    else{
        sem_wait(&sem_Division[t_id-1]);
    }
    
    int i, j;

    //正式步骤一：填充累加器
    auto start = std::chrono::high_resolution_clock::now();
    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
                for (int n = t_id; n < numangle; n+=ThreadNum)
                {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
        }
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    chrono::duration<double> duration = end - start;
    cout << "thread " << t_id <<" step1: " << duration.count() * 1000 << "ms" << endl;

    //为了满足前后步骤的依赖关系，必须让步骤一结束后的线程同步
    if(t_id==0)
    {
        for(int t = 0; t < ThreadNum-1; t++)
        {
            sem_wait(&sem_leader);
        }
        for(int t = 0; t < ThreadNum-1; t++)
        {
            sem_post(&sem_Elimination[t]);
        }
    }
    else{
        sem_post(&sem_leader);
        sem_wait(&sem_Elimination[t_id-1]);
    }


    //正式步骤二：寻找所有本地个最大值
    start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < numrho; r++)
        for (int n = t_id; n < numangle; n+=ThreadNum)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;
        
            if (accum[base] > threshold_ &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                {
                    // 在访问sort_buf之前加锁
                    pthread_mutex_lock(&sort_buf_mutex);

                    // 执行对sort_buf的操作
                    sort_buf.push_back(base);

                    // 在访问sort_buf之后解锁
                    pthread_mutex_unlock(&sort_buf_mutex);
                }
        }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "thread " << t_id <<" step2: " << duration.count() * 1000 << "ms" << endl;

    //等待第二部完全结束
    if(t_id==0)
    {
        for(int t = 0; t < ThreadNum-1; t++)
        {
            sem_wait(&sem_leader);
        }
        for(int t = 0; t < ThreadNum-1; t++)
        {
            sem_post(&sem_Elimination[t]);
        }
    }
    else{
        sem_post(&sem_leader);
        sem_wait(&sem_Elimination[t_id-1]);
        pthread_exit(NULL);
    }

    if(t_id==0)
    {
        //正式步骤三：从大到小排序
        start = std::chrono::high_resolution_clock::now();
        std::sort(sort_buf.begin(), sort_buf.end(), hough_cmp_gt(accum));
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "step3: " << duration.count() * 1000 << "ms" << endl;

        //正式步骤四：存储前 min(total,linesMax) 条直线
        start = std::chrono::high_resolution_clock::now();
        linesMax = std::min(linesMax, (int)sort_buf.size());
        cout<<"linesMax:"<<linesMax<<endl;
        double scale = 1. / (numrho + 2);

        for (i = 0; i < linesMax; i++)
        {
            Line line;
            int idx = sort_buf[i];
            int n = (int)(idx * scale) - 1;
            int r = idx - (n + 1) * (numrho + 2) - 1;
            line.rho = (r - (numrho - 1) * 0.5f) * rho;
            line.angle = static_cast<float>(min_theta) + n * theta;
            lines->push_back(line);
        }
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        cout << "step4: " << duration.count() * 1000 << "ms" << endl;

    }
    
    pthread_exit(NULL);
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

    // 初始化互斥锁
    pthread_mutex_init(&sort_buf_mutex, NULL);

    // 初始化所有信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < ThreadNum-1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    vector<Line> *lines = new vector<Line>;
    int linesMax = INT_MAX;
    pthread_t thread[ThreadNum];
    MyThreadParams params[ThreadNum];

    auto start = std::chrono::high_resolution_clock::now();
    for (int t_id = 0; t_id < ThreadNum; t_id++)
    {
        params[t_id].src = &image;
        params[t_id].lines = lines;
        params[t_id].type = 0;
        params[t_id].rho = 1;
        params[t_id].theta = pai / 180;
        params[t_id].threshold = 200;
        params[t_id].linesMax = linesMax;
        params[t_id].min_theta = 0;
        params[t_id].max_theta = pai;
        params[t_id].t_id = t_id;

        pthread_create(&thread[t_id], NULL, HoughLinesStandard, &params[t_id]);
    }

    for (int t_id = 0; t_id < ThreadNum; t_id++)
    {
        pthread_join(thread[t_id], NULL);
    }
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "total_time: " << duration.count() * 1000 << "ms" << endl;

    
    cout << "All the lines detected are below (rho+angle):\n";
    if(lines != nullptr && !lines->empty())
        for (auto iter = lines->begin(); iter != lines->end(); iter++)
            cout << iter->rho << " " << iter->angle << endl;
    else
        cout<<"lines equal 0!"<<endl;
    

   // 销毁所有信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < ThreadNum-1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }

    // 销毁互斥锁
    pthread_mutex_destroy(&sort_buf_mutex);
    
    
    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
    delete lines;
    return 0;
}