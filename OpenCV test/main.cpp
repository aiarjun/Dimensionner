#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat getEnergyImage(Mat& image){
    Mat image_blur,image_grey;
    int ddepth = CV_16S;
    Mat gradientX,gradientY;
    Mat absGradientX,absGradientY;
    Mat gradient;

    GaussianBlur(image,image_blur,Size(3,3),0,0,BORDER_DEFAULT);
    image_grey = image_blur;
    cvtColor(image_blur,image_grey,COLOR_BGR2GRAY);

//    Sobel(image_grey,gradientX,ddepth,1,0);
//    Sobel(image_grey,gradientY,ddepth,0,1);

    Scharr(image_grey,gradientX, ddepth,1,0);
    Scharr(image_grey,gradientY, ddepth,0,1);


    convertScaleAbs(gradientX,absGradientX);
    convertScaleAbs(gradientY,absGradientY);

    addWeighted(absGradientX,0.5,absGradientY,0.5,0,gradient);
    namedWindow("Gradient",WINDOW_NORMAL);
    imshow("Gradient",gradient);
    imwrite("images/scharr_gradient_smooth.jpg",gradient);
    waitKey(0);
    return gradient;
}


int main(int argc, char** argv) {
    int row = 485;
    int column = 1397;
    string imageName = "images/scenery.jpg";
    Mat image;
    image = imread(imageName,IMREAD_COLOR);
    Mat energyImage = getEnergyImage(image);

//    Vec3b intensity = image.at<Vec3b>(row,column);
//    float blue = intensity.val[0];
//    float green = intensity.val[1];
//    float red = intensity.val[2];
//
//    cout << "Blue:" << blue << endl;
//    cout << "Green:" << green  << endl;
//    cout << "Red:" << red << endl;





//    Mat image2 = image;
//    for(int x=0;x<100;x++){
//        for(int y=0;y<100;y++){
//            Vec3b color = image2.at<Vec3b>(y,x);
//            color[0] = 0;
//            color[1] = 255;
//            color[2] = 0;
//
//            image2.at<Vec3b>(y,x) = color;
//        }
//    }
//    imshow("New image",image2);
//    imwrite("new_image.jpg",image2);
    return 0;
}
