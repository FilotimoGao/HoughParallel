#include<iostream>
#include <stdlib.h>
#include<cmath>
#include<time.h>
#include <chrono>
#include <arm_neon.h>

using namespace std;

int main()
{
    int height, width, step;
    height = width = step =2048;
    int numangle = 180;
    int numrho = (int)((width + height) * 2 + 1 + 0.5);

    int size = height * width;
    int* image = new int[size];
    int* accum = new int[(numangle + 2)*(numrho + 2)]{0};

    for (int i = 0; i < size; i++)
    {
        image[i] = i % 10;
    }
    
    float* tabSin = new float[numangle];
    float* tabCos = new float[numangle];

    float ang = 0;
    float theta = 3.14 / 180;
    for (int n = 0; n < numangle; ang += theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang));    //做好tabSin数组，后面备查
        tabCos[n] = (float)(cos((double)ang));    //做好tabCos数组，后面备查
    }

    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
            {
                for (int n = 0; n < numangle; n+=4)
                {
                    //加载数组tabCos和tabSin的第n个元素及后续3个元素的值到向量中
                    float32x4_t tabCos4 = vld1q_f32(&tabCos[n]);
                    float32x4_t tabSin4 = vld1q_f32(&tabSin[n]);

                    //获取i和j的向量
                    float32x4_t j4 = vmovq_n_f32(j);
                    float32x4_t i4 = vmovq_n_f32(i);

                    //vcvtq_s32_f32将上一步得到的向量转换为int32x4_t类型的向量,四舍五入到最接近的整数
                    int32x4_t r4 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(j4, tabCos4), vmulq_f32(i4, tabSin4)));
                    r4 = vaddq_s32(r4, vdupq_n_s32((numrho - 1) / 2));

                    //给四个数组元素分别自增一
                    accum[(n + 1) * (numrho + 2) + (int)r4[0] +1] ++;
                    accum[(n + 2) * (numrho + 2) + (int)r4[0] +1] ++;
                    accum[(n + 3) * (numrho + 2) + (int)r4[0] +1] ++;
                    accum[(n + 4) * (numrho + 2) + (int)r4[0] +1] ++;
                    
                }
            }
        }
    }

    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ns" << std::endl;



    delete[] image;
    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;

    return 0;
}

