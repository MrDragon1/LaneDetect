#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <vector>
#include <time.h>
using namespace cv;
using namespace std;

Mat Gaussian_filter(Mat& img);
void cannyEdgeDetection(Mat img, Mat &result, double highThreshold, double lowThreshold);
void HoughP( Mat& image,float rho, float theta, int threshold,int lineLength, int lineGap,vector<Vec4i>& lines, int linesMax = INT_MAX);
void GenJson(vector<Mat> imgs,string path,int cost);
void Search(vector<Vec4i>lines, vector<Mat>& imgs,Mat src);
void ShrinkLines(vector<Vec4i>& lines);
void EdgeDetection(Mat img, Mat &result, double highThreshold);
Mat Preprocessing(Mat img);

int width , height;
vector<int> h_samples;
vector<string> sys_path;
int main() {
    //获取目录下所有测试样本路径
    char *filePath = "imgpath.txt";
    ifstream file;
    file.open(filePath,ios::in);
    if(!file.is_open()) return 0;
    std::string strLine;
    while(getline(file,strLine))
    {
        if(strLine.empty())
            continue;
        sys_path.push_back(strLine.substr(57) + "\\20.jpg");
    }
    for(int i = 160;i<=710;i+=10){
        h_samples.push_back(i);
    }

    clock_t start,end;
    //处理每一张图片
    for(string path:sys_path){
        start = clock();
        cout << path << endl;
        Mat img = imread("G:\\3-1\\DigitalImageProcessing\\final\\selected tesing data\\" + path);

        if(img.empty()) continue;
        Mat imgGray,imgBlur,imgCanny,result,imgHoughP;
        result = img.clone();
        width = img.cols;
        height = img.rows;
        /* To gray */
        imgGray = Mat::zeros(img.size(),CV_8UC1);
        for(int i = 0;i<height;i++)
        {
            for(int j = 0;j<width;j++)
            {
                imgGray.ptr<uchar>(i)[j] = 0.4 * img.at<Vec3b>(i,j)[0] + 0.6 * img.at<Vec3b>(i,j)[1] + 0 * img.at<Vec3b>(i,j)[2];
            }
        }

        imgBlur = Gaussian_filter(imgGray);
        cannyEdgeDetection(imgBlur, imgCanny,  150, 100);
        //EdgeDetection(imgBlur, imgCanny,  100, 50);
        /* set ROI */
        for(int i = 0;i<height;i++)
        {
            for(int j = 0;j<width;j++)
            {
                if(i<height*2/5||j<10||i>height-10||j>width-10)
                    imgCanny.ptr<uchar>(i)[j]=0;
            }
        }

        //imgCanny = Preprocessing(imgCanny);
        imshow("imgCanny",imgCanny);
        //waitKey();
        cvtColor(imgCanny,imgHoughP, COLOR_GRAY2BGR);

        vector<Vec4i> linesP;
        HoughP(imgCanny, 1, CV_PI/180, 100, 50, 300 ,linesP);

        /*
           TODO:选择出3或4条车道线 & 车道线尽头附近用曲线拟合
            以一定幅角向前搜索，优先选择距离前进方向最近的点作为下一个点
            如果前进方向一直没有点，一直搜到边界，然后用直线拟合
         */
        vector<Vec4i> lines;
        int count = 0;
        float ts = 0;
        for(auto p:linesP){
            for(auto r:linesP){
                float kp = (float)(p[1]-p[3])/(float)(p[0] - p[2]);
                float bp = p[1] - kp*p[0];

                float kr = (float)(r[1]-r[3])/(float)(r[0] - r[2]);
                float br = r[1] - kr*r[0];
                ts += sqrt(pow(kp-kr,2) + pow(bp-br,2));
            }
        }
        ts /= linesP.size()*(linesP.size()-1);
        ts *= 0.2;
        //cout << "ts: "<<ts<<endl;
        for(auto p:linesP){
            if(lines.empty()) {
                lines.push_back(p);
                continue;
            }
            float kp = (float)(p[1]-p[3])/(float)(p[0] - p[2]);
            float bp = p[1] - kp*p[0];
            bool flag = true;
            for(auto r:lines){
                float kr = (float)(r[1]-r[3])/(float)(r[0] - r[2]);
                float br = r[1] - kr*r[0];
                if(sqrt(pow(kp-kr,2) + pow(bp-br,2)) < ts){
                    flag = false;
                    break;
                }
            }
            if(flag) lines.push_back(p);
        }
        //cout<<"after ts: "<<lines.size()<<endl;
        vector<int>tmp;
        for(int i = 0;i<lines.size();i++){
            for(int j = i+1;j<lines.size();j++){
                float kp = (float)(lines[i][1]-lines[i][3])/(float)(lines[i][0] - lines[i][2]);
                float kr = (float)(lines[j][1]-lines[j][3])/(float)(lines[j][0] - lines[j][2]);
                if(abs(kp/kr - 1) < 0.2 || abs(kp) < 0.2){
                    bool flag = true;
                    for(int t:tmp){
                        if(t==j){
                            flag = false;
                        }
                    }
                    if(flag) tmp.push_back(j);
                }
            }
        }

        for(int t:tmp){
            if(t-count>=0&&t-count<lines.size()&&!lines.empty())
            {
                int s = t-count;
                lines.erase(lines.begin()+s);
                count++;
            }
        }

        cout<< "lines size: " <<lines.size()<<" del num: "<< count<<endl;

        vector<Mat> imgs;
        //将线段收缩到指定范围
        ShrinkLines(lines);
        //剩下部分采用向前搜索的方式
        Search(lines,imgs,imgCanny);

        //绘制路线
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            //Mat tempimg = Mat::zeros(img.size(),CV_8UC1);
            //line(tempimg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, LINE_AA);
            //imgs.push_back(tempimg);
            if(abs((double)(l[3]-l[1])/(double)(l[2]-l[0])) >= 0.2)
                line( imgHoughP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 2, LINE_AA);
        }
        end = clock();
        //生成json文件， 用于计算正确率
        GenJson(imgs,path,double(end-start)/CLOCKS_PER_SEC);

        addWeighted( img, 0.7, imgHoughP, 0.3, 0.0, result);
        imshow("output", result);

        waitKey(10);
    }
    return 0;
}


//高斯模糊
Mat Gaussian_filter(Mat& img)
{
    float gaussian_kernel[3][3] = {0.057118,0.12476,0.057118,0.12476,0.2725,0.12476,0.057118,0.12476,0.057118};
    Mat newImg = Mat::zeros(img.rows, img.cols, CV_64FC1);
    img.convertTo(img, CV_64FC1);
    for(int i = 1;i<height-1;i++)
    {
        for(int j = 1;j<width-1;j++)
        {
            for(int y = -1;y<=1;y++)
            {
                for(int x = -1;x<=1;x++)
                {
                    newImg.at<double>(i, j) += img.at<double>(i+y,j+x) * gaussian_kernel[y+1][x+1];
                }
            }
        }
    }
    return newImg;
}
void EdgeDetection(Mat img, Mat &result, double Threshold){
    Mat filterImg = img.clone();
    result = Mat::zeros(height, width, CV_64FC1);

    double gradX,gradY,grad;
    for(int i = 1 ; i < height-1; i++){
        for(int j = 1; j<width-1; j++){
            gradX = -filterImg.at<double>(i-1,j-1) - 2*filterImg.at<double>(i,j-1) - filterImg.at<double>(i+1,j-1)
                                    +filterImg.at<double>(i-1,j+1) + 2*filterImg.at<double>(i,j+1) + filterImg.at<double>(i+1,j+1);
            gradY = -filterImg.at<double>(i+1,j-1) - 2*filterImg.at<double>(i+1,j) - filterImg.at<double>(i+1,j+1)
                                    +filterImg.at<double>(i-1,j-1) + 2*filterImg.at<double>(i-1,j) + filterImg.at<double>(i-1,j+1);
            grad = sqrt(pow(gradX,2) + pow(gradY,2));
            if(grad > Threshold) result.at<double>(i, j) = 255;
        }
    }

    result.convertTo(result, CV_8UC1);
}



//canny边缘检测
void cannyEdgeDetection(Mat img, Mat &result, double highThreshold, double lowThreshold){
    Mat filterImg = img.clone();
    result = Mat::zeros(height, width, CV_64FC1);

    double gradX, gradY;
    Mat grad = Mat::zeros(height, width, CV_64FC1);
    Mat angle = Mat::zeros(height, width, CV_64FC1);
    Mat dir = Mat::zeros(height, width, CV_64FC1);

    for(int i = 1 ; i < height-1; i++){
        for(int j = 1; j<width-1; j++){
            gradX = -filterImg.at<double>(i-1,j-1) - 2*filterImg.at<double>(i,j-1) - filterImg.at<double>(i+1,j-1)
                                    +filterImg.at<double>(i-1,j+1) + 2*filterImg.at<double>(i,j+1) + filterImg.at<double>(i+1,j+1);
            gradY = -filterImg.at<double>(i+1,j-1) - 2*filterImg.at<double>(i+1,j) - filterImg.at<double>(i+1,j+1)
                                    +filterImg.at<double>(i-1,j-1) + 2*filterImg.at<double>(i-1,j) + filterImg.at<double>(i-1,j+1);
            grad.at<double>(i,j) = sqrt(pow(gradX,2) + pow(gradY,2));
            angle.at<double>(i, j) = atan(gradY/gradX);
            if(0 <= angle.at<double>(i,j) <= (CV_PI/4.0)){
                dir.at<double>(i, j) = 0;
            }
            else if(CV_PI/4.0 < angle.at<double>(i,j) <= (CV_PI/2.0)){
                dir.at<double>(i, j) = 1;
            }
            else if(-CV_PI/2.0 <= angle.at<double>(i,j) <= (-CV_PI/4.0)){
                dir.at<double>(i, j) = 2;
            }
            else if(-CV_PI/4.0 < angle.at<double>(i,j) < 0){
                dir.at<double>(i, j) = 3;
            }
        }
    }

    for(int i = 1 ; i<height-1; i++){
        for( int j = 1; j<width-1; j++){

            double tantheta;
            double Gp1, Gp2;
            tantheta = abs(tan(angle.at<double>(i,j)));
            switch ((int)dir.at<double>(i,j)) {
                case 0:
                    Gp1 = (1- tantheta) * grad.at<double>(i, j+1) + tantheta * grad.at<double>(i-1, j+1);
                    Gp2 = (1- tantheta) * grad.at<double>(i, j-1) + tantheta * grad.at<double>(i-1, j-1);
                    break;
                case 1:
                    Gp1 = (1- tantheta) * grad.at<double>(i-1, j) + tantheta * grad.at<double>(i-1, j+1);
                    Gp2 = (1- tantheta) * grad.at<double>(i+1, j) + tantheta * grad.at<double>(i-1, j-1);
                    break;
                case 2:
                    Gp1 = (1- tantheta) * grad.at<double>(i-1, j) + tantheta * grad.at<double>(i -1, j -1);
                    Gp2 = (1- tantheta) * grad.at<double>(i+1, j) + tantheta * grad.at<double>(i+1, j+1);
                    break;
                case 3:
                    Gp1 = (1- tantheta) * grad.at<double>(i, j-1) + tantheta * grad.at<double>(i -1, j -1);
                    Gp2 = (1- tantheta) * grad.at<double>(i, j+1) + tantheta * grad.at<double>(i+1, j+1);
                    break;
                default:
                    break;
            }

            if(grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2){
                if(grad.at<double>(i, j) >= highThreshold){
                    grad.at<double>(i, j) = highThreshold;
                    result.at<double>(i, j) = 255;
                }
                else if(grad.at<double>(i, j) < lowThreshold){
                    grad.at<double>(i, j) = 0;
                }
                else{
                    grad.at<double>(i, j) = lowThreshold;
                }

            }
            else{
                grad.at<double>(i, j) = 0;
            }
        }
    }

    for(int i = 1 ; i<height-1; i++){
        for(int j = 1; j<width-1; j++){
            //低阈值点周围没有高阈值点则放弃
            if(grad.at<double>(i, j) == lowThreshold){
                for(int i1 = -1; i1 < 2; i1++){
                    for(int j1 = -1; j1<2; j1++){
                        if(grad.at<double>(i + i1,j + j1) == highThreshold){
                            result.at<double>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    result.convertTo(result, CV_8UC1);
}

//概率霍夫变换
void HoughP(Mat& image,float rho, float theta, int threshold,int lineLength, int lineGap,vector<Vec4i>& lines, int linesMax)
{
    int width, height;
    int numangle, numrho;
    int r, n, count;
    float irho = 1 / rho;
    const int shift = 16;
    width = image.cols;
    height = image.rows;

    numangle = cvRound(CV_PI / theta);
    numrho = cvRound(((width + height) * 2 + 1) / rho);

    vector<Vec2i> seq;
    vector<vector<int>> accum;
    vector<vector<unsigned int>> mask;

    vector<int> tmp(numrho);
    accum.resize(numangle,tmp);

    vector<unsigned int> temp(width);
    mask.resize(height,temp);

    //保存所有边缘点到seq
    for(int i = 0;i<height;i++){
        const uchar* data = image.data + i*image.step;
        for(int j = 0;j<width;j++){
            if(data[j]){
                mask[i][j] = 1;
                seq.push_back(Point(j,i));
            }
            else mask[i][j] = 0;
        }
    }

    count = seq.size();
    const int w = 3;
    //随机处理所有的边缘点
    for(; count > 0; count--)
    {
        int index = rand() % count;
        int max_val = threshold-1, max_n = 0;
        Point point = seq[index];
        Point line_end[2] = {{0,0}, {0,0}};
        float a, b;
        int i, j, k, x0, y0, dx0, dy0, xflag;
        int good_line;


        i = point.y;
        j = point.x;

        seq[index] = seq[count-1];

        if(!mask[i][j])continue;

        for( n = 0; n < numangle; n++){
            r = cvRound( j * (float)(cos(n*theta) * irho) + i * (float)(sin(n*theta) * irho)) + (numrho - 1) / 2;
            int val = ++accum[n][r];
            if(max_val < val){
                max_val = val;
                max_n = n;
            }
        }
        if(max_val < threshold) continue;

        a = -(float)(sin(max_n*theta) * irho);
        b = (float)(cos(max_n*theta) * irho);
        x0 = j;
        y0 = i;
        if(fabs(a) > fabs(b)){
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound(b*(1 << shift)/fabs(a));
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else{
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound(a*(1 << shift)/fabs(b));
            x0 = (x0 << shift) + (1 << (shift-1));
        }
        //搜索直线的两个端点
        for(k = 0; k < 2; k++){
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;
            if(k == 1){
                dx = -dx;
                dy = -dy;
            }
            while(1){
                int i1, j1;
                j1 = xflag?x:x>>shift;
                i1 = xflag?y>>shift:y;
                if(j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
                    break;
                //新的端点
                bool flag = false;
                for(int ii = -w;ii<=w;ii++){
                    for(int jj = -w;jj<=w;jj++){
                        if(i1+ii>=0&&i1+ii<height&&j1+jj>=0&&j1+jj<width&&mask[i1+ii][j1+jj]) flag = true;
                    }
                }
                if(flag){
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if(++gap > lineGap)
                    break;
                x += dx;
                y += dy;
            }
        }
        good_line = abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    abs(line_end[1].y - line_end[0].y) >= lineLength;

        //再次搜索端点，目的是更新累加器矩阵和更新掩码矩阵，以备下一次循环使用
        for(k = 0; k < 2; k++){
            int x = x0, y = y0, dx = dx0, dy = dy0;
            if(k == 1){
                dx = -dx;
                dy = -dy;
            }
            while(1){
                int i1, j1;
                j1 = xflag?x:x>>shift;
                i1 = xflag?y>>shift:y;

                bool flag = false;
                for(int ii = -w;ii<=w;ii++){
                    for(int jj = -w;jj<=w;jj++){
                        if(i1+ii>=0&&i1+ii<height&&j1+jj>=0&&j1+jj<width&&mask[i1+ii][j1+jj]) flag = true;
                    }
                }
                if(flag){
                    if(good_line){
                        for(n = 0; n < numangle; n++){
                            r = cvRound( j1 * (float)(cos(n*theta) * irho) + i1 * (float)(sin(n*theta) * irho)) + (numrho - 1) / 2;
                            accum[n][r]--;
                        }
                    }
                    mask[i1][j1] = 0;
                }
                if(i1 == line_end[k].y && j1 == line_end[k].x)
                    break;
                x += dx;
                y += dy;
            }
        }

        if(good_line){
            Vec4i lr = {line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y};
            lines.push_back(lr);
            if(lines.size() >= linesMax)
                return;
        }
    }
}

//生成json文件
void GenJson(vector<Mat> imgs,string path,int cost){
    std::string file = "{\"lanes\": [";

    bool flag = false;
    for(Mat img:imgs){
        vector<int> line;
        for(int y:h_samples){
            int meanx = 0;
            int count = 0;
            for(int x = 0;x<width;x++){
                if(img.ptr<uchar>(y)[x]==255){
                    meanx += x;
                    count++;
                }
            }
            if(count == 0)line.push_back(-2);
            else line.push_back(meanx/count);
        }
        stringstream ss;
        std::string temp;
        copy(line.begin(),line.end(),ostream_iterator<int>(ss,", "));
        temp = ss.str();
        temp.pop_back();
        temp.pop_back();
        file += flag?",[":"[" ;
        file += temp + "]";
        flag = true;
        temp.clear();
    }
    stringstream ss;
    std::string temp;
    copy(h_samples.begin(),h_samples.end(),ostream_iterator<int>(ss,", "));
    temp = ss.str();
    temp.pop_back();
    temp.pop_back();
    file += "], \"h_samples\": [" + temp + "]";
    path.replace(path.find('\\'), 1, "/");
    path.replace(path.find('\\'), 1, "/");
    file += ", \"raw_file\": \"clips/" + path + "\", \"run_time\":" + to_string(cost) + " }";


    std::string file_name = "test.json";
    std::ofstream file_writer(file_name, std::ios_base::app|std::ios_base::out);
    file_writer << file << std::endl;
    file_writer.close();
}

//向前搜索车道线
void Search(vector<Vec4i>lines, vector<Mat>& imgs,Mat src){
    int th_y = 10;//看见前方10像素之后的区域
    Mat showimg = Mat::zeros(src.size(),CV_8UC1);

    for(auto line:lines){
        Mat img = Mat::zeros(src.size(),CV_8UC1);
        int rect_w = 10,rect_h = 10;
        int x0 = line[0],y0 = line[1],x1 = line[2],y1 = line[3];
        if(abs(y1-y0)<50){
            cout<<"continue"<<endl;
            continue;
        }
        float r = sqrt(pow(x0-x1,2) + pow(y0-y1,2));
        float theta = atan((float)(y0-y1)/(float)(x0-x1));
//        if(abs(theta)<CV_PI/18){
//            cout<<"continue"<<endl;
//            continue;
//        }
        float dx = -1/tan(theta);
        int pos_x,pos_y;
        pos_x = y1<y0 ? x1:x0;
        pos_y = y1<y0 ? y1:y0;
        int next_x,next_y;
        while(pos_y > height / 3){
            //TODO:前进方向上筛选像素点最多的区域
            //TODO:降低给定直线的高度到height*2/3
            next_y = pos_y + th_y;
            next_x = pos_x + dx*th_y;


            bool flag = true;
            int max_index=-99,index=0,max = 0;
            for(index = -10;index<=10;index++){
                int tmp_x = next_x + index*rect_w;
                int count = 0;
                for(int shift_y = -rect_h;shift_y<rect_h;shift_y++){
                    for(int shift_x = -rect_w;shift_x<rect_w;shift_x++){
                        if(tmp_x + shift_x >= 0 && tmp_x + shift_x < width && src.ptr<uchar>(next_y + shift_y)[tmp_x + shift_x] == 255){
                            count++;
                        }
                    }
                }
                if(count > max){
                    max = count;
                    max_index = index;
                }
            }
            //区域内有大于阈值个点
            if(max > rect_w*rect_h*0.5){
                //cout<<"test"<<endl;
               // next_x += max_index*rect_w;
                theta = atan((float)(-th_y)/(float)(next_x - pos_x));
                dx = -1/tan(theta);
            }


//            int search_w = 50;
//            for(int search_y = pos_y;search_y>pos_y - th_y;search_y--){
//                for(int search_x = pos_x-search_w;search_x <= pos_x+search_w;search_x++){
//                    if(search_x>=0 && search_x<width && src.ptr<uchar>(search_y)[search_x] == 255){
//                        cv::circle(img,Point(search_x,search_y),1,Scalar(255),2);
//                        cv::circle(showimg,Point(search_x,search_y),1,Scalar(255),2);
//                    }
//                }
//            }

//            int tmp_x = 0;
//            int tmp_count = 0;
//            for(int search_x = pos_x-search_w;search_x <= pos_x+search_w;search_x++){
//                if(search_x>=0 && search_x<width && img.ptr<uchar>(pos_y - th_y - 1)[search_x] == 255){
//                    tmp_x += search_x;
//                    tmp_count++;
//                }
//            }
//            if(tmp_count>0){
//                tmp_x /= tmp_count;
//                next_x = tmp_x;
//                theta = atan((float)(-th_y)/(float)(next_x - pos_x));
//                dx = -1/tan(theta);
//            }

            cv::line(img, Point(pos_x,pos_y), Point(next_x, pos_y-th_y), Scalar(255), 2, LINE_AA);
            cv::line(showimg, Point(pos_x,pos_y), Point(next_x, pos_y-th_y), Scalar(255), 2, LINE_AA);

            pos_x = next_x;
            pos_y -= th_y;
        }

        cv::line(img, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(255), 2, LINE_AA);
        cv::line(showimg, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(155), 2, LINE_AA);
        imgs.push_back(img);
    }
    //imshow("img_lane_before",showimg);
    //只有一半有车道线时将图像镜像
    bool empty_l = true,empty_r = true;
    for(int i = 0;i<height && empty_l && empty_r;i++){
        for(int j = 0;j<width/3 && empty_l && empty_r;j++){
            if(showimg.ptr<uchar>(i)[j] > 0) empty_l = false;
            if(showimg.ptr<uchar>(i)[width - j + 1] > 0) empty_r = false;
        }
    }
    if((empty_l&&empty_r)||imgs.size() <= 1){;
        const int size = imgs.size();
        for(int index = 0;index < size;index++){
            Mat pic = imgs[index];
            Mat newpic = Mat::zeros(pic.size(),CV_8UC1);
            for(int i = 0;i < height;i++){
                for(int j = 0;j < width;j++){
                    if(pic.ptr<uchar>(i)[j] == 255){
                       newpic.ptr<uchar>(i)[width - j - 1] = 255;
                       showimg.ptr<uchar>(i)[width - j - 1] = 255;
                    }
                }
            }
            imgs.push_back(newpic);
        }
    }
    imshow("img_lane",showimg);

    waitKey(10);
}

//收缩所有线段到指定范围
void ShrinkLines(vector<Vec4i>& lines){
    for(int i = 0;i<lines.size();i++){
        Vec4i line = lines[i];
        int x0 = line[1]>line[3]?line[0]:line[2],y0 = line[1]>line[3]?line[1]:line[3],x1 = line[1]<=line[3]?line[0]:line[2],y1 = line[1]<=line[3]?line[1]:line[3];
        if(x0 == x1)continue;
        float k = (float)(y1-y0)/(float)(x1-x0);
        Vec4i newline(x0,y0,x1,y1);

        //向下收缩到height-10
        int pos_x = x0,pos_y = y0;
        float dx = 1/k,dy = 1;
        while(pos_x>0 && pos_x<width-1 && pos_y<height-10){
            pos_x = x0 + dx * dy;
            dy++;
            pos_y++;
        }
        newline[0] = pos_x;
        newline[1] = pos_y;

        //向上收缩到height*2/3
        pos_x = x1;
        pos_y = y1;
        dy = 1;
        while(pos_x>0 && pos_x<width-1 && pos_y<height/3){
            pos_x = x1 + dx * dy;
            dy++;
            pos_y++;
        }
        newline[2] = pos_x;
        newline[3] = pos_y;
        lines[i] = newline;
    }
}

//膨胀、腐蚀操作去噪点
Mat Preprocessing(Mat img){
    int rect_w = 1,rect_h = 1;
    Mat tmp = Mat::zeros(img.size(),CV_8UC1);
    Mat tmp2 = Mat::zeros(img.size(),CV_8UC1);
    Mat res = Mat::zeros(img.size(),CV_8UC1);

//    //腐蚀
//    for(int i = rect_h;i<height - rect_h;i++){
//        for(int j = rect_w;j<width - rect_w;j++){
//            bool flag = true;
//            for(int ii = -rect_h;ii<rect_h;ii++){
//                for(int jj = -rect_w;jj<rect_w;jj++){
//                    if(img.ptr<uchar>(i+ii)[j+jj] != 255){
//                        flag = false;
//                    }
//                }
//            }
//            if(flag) tmp.ptr<uchar>(i)[j] = 255;
//        }
//    }
//    //膨胀
//    for(int i = rect_h;i<height - rect_h;i++){
//        for(int j = rect_w;j<width - rect_w;j++){
//            if(tmp.ptr<uchar>(i)[j] == 255){
//                for(int ii = -rect_h;ii<rect_h;ii++){
//                    for(int jj = -rect_w;jj<rect_w;jj++){
//                        tmp2.ptr<uchar>(i+ii)[j+jj] = 255;
//                    }
//                }
//            }
//        }
//    }

    rect_w = 1;
    rect_h = 1;
    for(int i = rect_h;i<height - rect_h;i++){
        for(int j = rect_w;j<width - rect_w;j++){
            if(img.ptr<uchar>(i)[j] == 255){
                for(int ii = -rect_h;ii<rect_h;ii++){
                    for(int jj = -rect_w;jj<rect_w;jj++){
                        res.ptr<uchar>(i+ii)[j+jj] = 255;
                    }
                }
            }
        }
    }
    return res;
}