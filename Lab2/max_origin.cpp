#include<iostream>
#include <stdlib.h>
#include<cmath>
#include<time.h>
#include <chrono>
#include <arm_neon.h>
#include <vector>

using namespace std;

int main()
{
    int height, width;
    height = width = 1024;
    int numangle = 180;
    int numrho = (int)((width + height) * 2 + 1 + 0.5);
    int threshold=100;

    int size = height * width;
    int* image = new int[size];
    int* accum = new int[(numangle + 2)*(numrho + 2)]{0};

    for (int i = 0; i < size; i++)
    {
        image[i] = i % 10;
    }

    int step = 1024;
    
    float* tabSin = new float[numangle];
    float* tabCos = new float[numangle];

    float ang = 0;
    float theta = 3.14 / 180;
    for (int n = 0; n < numangle; ang += theta, n++)
    {
        tabSin[n] = (float)(sin((double)ang));    //做好tabSin数组，后面备查
        tabCos[n] = (float)(cos((double)ang));    //做好tabCos数组，后面备查
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (image[i * step + j] != 0)
            {
                for (int n = 0; n < numangle; n++)
                {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
            }
        }
    }

    vector<int> sort_buf1;
    vector<int> sort_buf2;
    
    auto start1 = std::chrono::steady_clock::now();
    
    for(int n = 0; n < numangle; n++ )
        for(int r = 0; r < numrho; r++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf1.push_back(base);
        }
    
    auto end1 = std::chrono::steady_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();
    std::cout << "Execution time (n-r): " << duration1 << " ns" << std::endl;




    auto start2 = std::chrono::steady_clock::now();
    
    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf2.push_back(base);
        }
    
    auto end2 = std::chrono::steady_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
    std::cout << "Execution time (r-n): " << duration2 << " ns" << std::endl;
    

    delete[] image;
    delete[] accum;
    delete[] tabSin;
    delete[] tabCos;

    return 0;
}
