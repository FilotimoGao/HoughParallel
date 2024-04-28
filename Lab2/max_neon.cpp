#include <iostream>
#include <chrono>
#include <vector>
#include <math.h>
#include <arm_neon.h>

int main()
{
    int height, width, step;
    height = width = step = 1024;
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

    std::vector<int> sort_buf;

    auto start = std::chrono::steady_clock::now();

    int32x4_t threshold_vector = vdupq_n_s32(threshold);
    int32x4_t increment_vector = vdupq_n_s32(1); 
    int32x4_t numrho_plus_2 = vdupq_n_s32(numrho+2);

    for (int n = 0; n < numangle; n++ )
    {
        for (int r = 0; r < numrho; r += 4)
        {
            int base = (n+1) * (numrho+2) + r+1;

            int32x4_t accum_base = vld1q_s32(&accum[base]);
            int32x4_t accum_base_minus_1 = vld1q_s32(&accum[base - 1]);
            int32x4_t accum_base_plus_1 = vld1q_s32(&accum[base + 1]);
            int32x4_t accum_base_minus_numrho_minus_2 = vld1q_s32(&accum[base - numrho - 2]);
            int32x4_t accum_base_plus_numrho_plus_2 = vld1q_s32(&accum[base + numrho + 2]);

            uint32x4_t mask1 = vcgtq_s32(accum_base, threshold_vector);
            uint32x4_t mask2 = vcgtq_s32(accum_base, accum_base_minus_1);
            uint32x4_t mask3 = vcgeq_s32(accum_base, accum_base_plus_1);
            uint32x4_t mask4 = vcgtq_s32(accum_base, accum_base_minus_numrho_minus_2);
            uint32x4_t mask5 = vcgeq_s32(accum_base, accum_base_plus_numrho_plus_2);

            uint32x4_t result_mask = vandq_u32(vandq_u32(mask1, mask2), vandq_u32(mask3, vandq_u32(mask4, mask5)));

            int32x2_t result_mask_low = vget_low_s32(vreinterpretq_s32_u32(result_mask));
            int32x2_t result_mask_high = vget_high_s32(vreinterpretq_s32_u32(result_mask));

            int32_t mask_low = vget_lane_s32(result_mask_low, 0);
            int32_t mask_high = vget_lane_s32(result_mask_high, 0);

            if (mask_low)
            {
                sort_buf.push_back(base);
            }
            if (mask_high)
            {
                sort_buf.push_back(base + 1);
            }
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " ns" << std::endl;

    return 0;
}
