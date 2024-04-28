#include<iostream>
#include <stdlib.h>
#include<cmath>
#include<time.h>
#include <chrono>
#include <emmintrin.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2


using namespace std;

int main()
{
    int height, width, step;
    height = width = step = 128;
    int numangle = 180;
    int numrho = (int)((width + height) * 2 + 1 + 0.5);

    int size = height * width;
    int* image = new int[size];
    int* accum = new int[(numangle + 2) * (numrho + 2)] {0};

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
                for (int n = 0; n < numangle; n += 4)
                {
                    //加载数组tabCos和tabSin的第n个元素及后续3个元素的值到SSE寄存器中
                    __m128 tabCos4 = _mm_loadu_ps(&tabCos[n]);
                    __m128 tabSin4 = _mm_loadu_ps(&tabSin[n]);

                    //获取i和j的SSE寄存器
                    __m128 j4 = _mm_set1_ps((float)j);
                    __m128 i4 = _mm_set1_ps((float)i);

                    //执行向量化计算操作
                    __m128 result = _mm_add_ps(_mm_mul_ps(j4, tabCos4), _mm_mul_ps(i4, tabSin4));
                    __m128i r4 = _mm_cvtps_epi32(result);
                    r4 = _mm_add_epi32(r4, _mm_set1_epi32((numrho - 1) / 2));

                    //将结果存储回内存
                    accum[(n + 1) * (numrho + 2) + _mm_extract_epi32(r4, 0) + 1] ++;
                    accum[(n + 2) * (numrho + 2) + _mm_extract_epi32(r4, 1) + 1] ++;
                    accum[(n + 3) * (numrho + 2) + _mm_extract_epi32(r4, 2) + 1] ++;
                    accum[(n + 4) * (numrho + 2) + _mm_extract_epi32(r4, 3) + 1] ++;
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
