#include<iostream>
#include <stdlib.h>
#include<cmath>
#include<time.h>
#include <chrono>

using namespace std;

int main()
{
    int height, width;
    height = width = 512;
    int numangle = 180;
    int numrho = (int)((width + height) * 2 + 1 + 0.5);

    int size = height * width;
    int* image = new int[size];
    int* accum = new int[(numangle + 2)*(numrho + 2)]{0};

    for (int i = 0; i < size; i++)
    {
        image[i] = i % 10;
    }

    int step = 512;
    
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
                for (int n = 0; n < numangle; n++)
                {
                    int r = (int)(j * tabCos[n] + i * tabSin[n] + 0.5);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
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

