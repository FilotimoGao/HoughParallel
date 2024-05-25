/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
static void
HoughLinesStandard( InputArray src, OutputArray lines, int type,
                    float rho, float theta,
                    int threshold, int linesMax,
                    double min_theta, double max_theta )
{
    CV_CheckType(type, type == CV_32FC2 || type == CV_32FC3, "Internal error");

    Mat img = src.getMat();

    int i, j;
    float irho = 1 / rho;

    CV_Assert( img.type() == CV_8UC1 );
    CV_Assert( linesMax > 0 );

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;

    int max_rho = width + height;
    int min_rho = -max_rho;

    CV_CheckGE(max_theta, min_theta, "max_theta must be greater than min_theta");

    int numangle = computeNumangle(min_theta, max_theta, theta);
    int numrho = cvRound(((max_rho - min_rho) + 1) / rho);

    Mat _accum = Mat::zeros( (numangle+2), (numrho+2), CV_32SC1 );
    std::vector<int> _sort_buf;
    AutoBuffer<float> _tabSin(numangle);
    AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum.ptr<int>();
    float *tabSin = _tabSin.data(), *tabCos = _tabCos.data();

    // create sin and cos table
    createTrigTable( numangle, min_theta, theta,
                     irho, tabSin, tabCos);

    // stage 1. fill accumulator
    for( i = 0; i < height; i++ )
        for( j = 0; j < width; j++ )
        {
            if( image[i * step + j] != 0 )
                for(int n = 0; n < numangle; n++ )
                {
                    int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }

    // stage 2. find local maximums
    findLocalMaximums( numrho, numangle, threshold, accum, _sort_buf );

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1./(numrho+2);

    lines.create(linesMax, 1, type);
    Mat _lines = lines.getMat();
    for( i = 0; i < linesMax; i++ )
    {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        if (type == CV_32FC2)
        {
            _lines.at<Vec2f>(i) = Vec2f(line.rho, line.angle);
        }
        else
        {
            CV_DbgAssert(type == CV_32FC3);
            _lines.at<Vec3f>(i) = Vec3f(line.rho, line.angle, (float)accum[idx]);
        }
    }
}