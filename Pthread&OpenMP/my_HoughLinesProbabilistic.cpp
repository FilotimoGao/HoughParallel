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

    Array(int h, int w, int s):height(h),width(w),step(s)
    {
        array = new int[height * width];
    }
    ~Array()
    {
        delete[] array;
    }
};

struct myPoint
{
public:
    int x;
    int y;
    myPoint(int x, int y):x(x),y(y){}
};

//代表线段
class Line
{
public:
    myPoint p1,p2;
};


void
HoughLinesProbabilistic( const Array &src,
    float rho, float theta, int threshold,
    int lineLength, int lineGap,
    std::vector<Line>& lines, int linesMax )
{
    myPoint pt(0,0);
    float irho = 1 / rho;

    srand(time(0));
    int rng;

    int width = src.width;
    int height = src.height;
    
    //需要遍历的角度数量
    int numangle = (int)(pai / theta) + 1;
    if (numangle > 1 && fabs(pai - (numangle - 1) * theta) < theta / 2)
        --numangle;
    //需要遍历的步数
    int numrho = (int)(((width + height) * 2 + 1) / rho + 0.5);

    //累加器
    int *accum = new int[numangle*numrho];
    //掩膜
    int *mask = new int[height*width];
    vector<float> trigtab(numangle*2);

    for( int n = 0; n < numangle; n++ )
    {
        trigtab[n*2] = (float)(cos((double)n*theta) * irho);
        trigtab[n*2+1] = (float)(sin((double)n*theta) * irho);
    }
    const float* ttab = &trigtab[0];
    vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for( pt.y = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] )
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();
}