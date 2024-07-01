#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <climits>
#include <random>
#include <time.h>
#define pai 3.14159265358979323846

using namespace std;


//代表输入数组，由于可能是二维的，所有有高和宽
class Array
{
public:
    int height, width, step;
    int* array;

    Array(int h, int w, int s) :height(h), width(w), step(s)
    {
        array = new int[height * width];
    }
    ~Array()
    {
        delete[] array;
    }
};

struct Point
{
    int x;
    int y;
    Point() {}
    Point(int x, int y) :x(x), y(y) {}
};

//代表线段
struct Line
{
    Point p1, p2;
    Line() {}
    Line(int x1, int y1, int x2, int y2) :p1(x1, y1), p2(x2, y2) {}
};

//代表直线
struct Line_ra
{
    float rho;
    float angle;
    Line_ra(){}
    Line_ra(float r,float a):rho(r),angle(a){}
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
HoughLinesStandard(const Array& src, vector<Line_ra>& lines, int type,
    float rho, float theta,
    int threshold, int& linesMax,
    double min_theta, double max_theta)
{
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

    float ang = static_cast<float>(min_theta);
   
    int i, j;
    std::vector<int> sort_buf;
    vector<Point> nzloc;
    Point pt(0, 0);

    //正式步骤一：填充累加器
    auto start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < numangle; ang += (float)theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }

    for (pt.y = 0; pt.y < height; pt.y++)
    {
        const int* data = &image[pt.y * step];
        for (pt.x = 0; pt.x < width; pt.x++)
        {
            if (data[pt.x])
                nzloc.push_back(pt);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    chrono::duration<double> duration = end - start;
    cout << "step1_1: " << duration.count() * 1000 << "ms" << endl;


    //拆分步骤一
    start = std::chrono::high_resolution_clock::now();
    int size = (int)nzloc.size();
    for (int i = 0; i < size; i++)
    {
        int* adata = accum;
        adata += (numrho + 2);
        for (int n = 0; n < numangle; n++, adata += (numrho + 2))
        {
            int r = (int)(nzloc[i].x * tabCos[n] + nzloc[i].y * tabSin[n] + 0.5);
            r += (numrho - 1) / 2;
            adata[r + 1]++;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "step1_2: " << duration.count() * 1000 << "ms" << endl;


    //正式步骤二：寻找所有本地个最大值
    start = std::chrono::high_resolution_clock::now();
    
    for (int n = 0; n < numangle; n++)
        for (int r = 0; r < numrho; r++)
        {
            int base = (n + 1) * (numrho + 2) + r + 1;

            if (accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                sort_buf.push_back(base);
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
        Line_ra line;
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

    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;
}






void
HoughLinesProbabilistic(const Array& src,
    float rho, float theta, int threshold,
    int lineLength, int lineGap,
    std::vector<Line_ra>& lines, int& linesMax)
{
    Point pt(0, 0);
    float irho = 1 / rho;
    int* image = src.array;

    srand(time(0));

    int step = src.step;
    int width = src.width;
    int height = src.height;

    //需要遍历的角度数量
    int numangle = (int)(pai / theta + 0.5);
    //需要遍历的步数
    int numrho = (int)(((width + height) * 2 + 1) / rho + 0.5);

    //累加器
    int* accum = new int[numangle * numrho] {0};
    //掩码矩阵
    int* mask = new int[height * width] {0};
    float* trigtab = new float[numangle * 2];

    //-------------------------------------------------------------------开始计时
    auto start = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < numangle; n++)
    {
        trigtab[n * 2] = (float)(cos((double)n * theta) * irho);
        trigtab[n * 2 + 1] = (float)(sin((double)n * theta) * irho);
    }
    const float* ttab = &trigtab[0];
    int* mdata0 = mask;
    vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for (pt.y = 0; pt.y < height; pt.y++)
    {
        //提取出输入图像和掩码矩阵的每行地址指针  
        const int* data = &image[pt.y * step];
        int* mdata = &mask[pt.y * step];
        for (pt.x = 0; pt.x < width; pt.x++)
        {
            if (data[pt.x])
            {
                mdata[pt.x] = (int)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    chrono::duration<double> duration = end - start;
    cout << "step1: " << duration.count() * 1000 << "ms" << endl;
    //cout << "count:" << count << endl;

    // stage 2. process all the points in random order
    start = std::chrono::high_resolution_clock::now();
    for (; count > 0; count--)
    {
        // choose random point out of the remaining ones
        int idx = rand() % count;
        int max_val = threshold - 1, max_n = 0;
        float max_r = 0;
        Point point = nzloc[idx];
        //cout << "point " << idx << ": (" << point.x << ", " << point.y << ")" << endl;
        Point line_end[2] = { {0,0},{0,0} };
        float a, b;

        int* adata = accum;

        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count - 1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if (!mdata0[i * width + j])
            continue;

        // update accumulator, find the most probable line
        //找到经过该点的累加器数值最大的直线方向
        for (int n = 0; n < numangle; n++, adata += numrho)
        {
            int r = (int)(j * ttab[n * 2] + i * ttab[n * 2 + 1] + 0.5);
            r += (numrho - 1) / 2;
            //cout << "r: " << r << endl;
            int val = ++adata[r];
            if (max_val < val)
            {
                max_val = val;
                max_n = n;
                max_r = r;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if (max_val < threshold)
            continue;

        Line_ra line;
        line.rho = (max_r - (numrho - 1) * 0.5f) * rho;
        line.angle = max_n * theta;
        lines.push_back(line);

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n * 2 + 1];
        b = ttab[max_n * 2];
        x0 = j;
        y0 = i;

        if (fabs(a) > fabs(b))
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = (int)(b * (1 << shift) / fabs(a) + 0.5);
            y0 = (y0 << shift) + (1 << (shift - 1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = (int)(a * (1 << shift) / fabs(b) + 0.5);
            x0 = (x0 << shift) + (1 << (shift - 1));
        }


        for (k = 0; k < 2; k++)
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if (k > 0)
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetic,
            // stop at the image border or in case of too big gap
            for (;; x += dx, y += dy)
            {
                int* mdata;
                int i1, j1;

                if (xflag)
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1 * width + j1;
                //cout << "i1,j1" << i1 << " " << j1 << endl;

                //如果到达了图像的边界，停止位移，退出循环  
                if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    break;
                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if (*mdata)
                {
                    adata = accum;
                    for (int n = 0; n < numangle; n++, adata += numrho)
                    {
                        int r = (int)(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1] + 0.5);
                        r += (numrho - 1) / 2;
                        adata[r]--;
                    }
                    *mdata = 0;
                }

                
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "step2: " << duration.count() * 1000 << "ms" << endl;

    delete[] accum;
    delete[] mask;
    delete[] trigtab;
}


int main()
{
    //ifstream file("edges2.txt");
    ifstream file("edges.txt");

    if (!file.is_open()) {
        cout << "file_open_error!" << endl;
        return 0;
    }

    //Array image(534, 800, 800);
    Array image(800, 800, 800);


    char pixel;
    int temp = 0;
    while (file >> pixel) {
        if (pixel != '\n') {
            image.array[temp++] = (int)pixel - 48;
        }
    }
    file.close();


    vector<Line_ra> lines;
    int linesMax = INT_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    HoughLinesProbabilistic(image, 1, pai / 180, 200, 30, 10, lines, linesMax);
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    chrono::duration<double> duration = end - start;
    cout << "total time(Probabilistic): " << duration.count() * 1000 << "ms" << endl;


    cout << "linesMax: " << lines.size() << endl;
    //cout << "All the lines detected are below (x1,y1,x2,y2):\n";
    cout << "All the lines detected are below (rho+angle):\n";
    for (auto iter = lines.begin(); iter != lines.end(); iter++)
    {
        cout << iter->rho << " " << iter->angle << endl;
    }

    cout << endl;

    vector<Line_ra> lines_;
    int linesMax_ = INT_MAX;
    start = std::chrono::high_resolution_clock::now();
    HoughLinesStandard(image, lines_, 0, 1, pai / 180, 200, linesMax_, 0, pai);
    end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    duration = end - start;
    cout << "total time(Standard): " << duration.count() * 1000 << "ms" << endl;

    cout << "linesMax: " << lines_.size() << endl;
    cout << "All the lines detected are below (rho+angle):\n";
    for (auto iter = lines_.begin(); iter != lines_.end(); iter++)
    {
        cout << iter->rho << " " << iter->angle << endl;
    }
    
    return 0;
}

