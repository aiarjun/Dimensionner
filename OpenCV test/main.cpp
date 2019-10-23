#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat getEnergyImage(Mat& image){
    Mat image_blur,image_grey;
    int ddepth = CV_16S;
    Mat gradientX,gradientY;
    Mat absGradientX,absGradientY;
    Mat gradient;
    Mat energyImage;

    GaussianBlur(image,image_blur,Size(3,3),0,0,BORDER_DEFAULT);
    image_grey = image_blur;
    cvtColor(image_blur,image_grey,COLOR_BGR2GRAY);

    Sobel(image_grey,gradientX,ddepth,1,0);
    Sobel(image_grey,gradientY,ddepth,0,1);

//    Scharr(image_grey,gradientX, ddepth,1,0);
//    Scharr(image_grey,gradientY, ddepth,0,1);


    convertScaleAbs(gradientX,absGradientX);
    convertScaleAbs(gradientY,absGradientY);

    addWeighted(absGradientX,0.5,absGradientY,0.5,0,gradient);

    gradient.convertTo(energyImage,CV_64F,1.0/255.0);


    namedWindow("Energy image",WINDOW_NORMAL);
    imshow("Energy image",energyImage);waitKey(0);

    return energyImage;
}


Mat getCumulativeEnergyMap(Mat& energyImage){
    int rowsize = energyImage.rows;
    int colsize = energyImage.cols;

    double upperLeftCumulativeEnergy,upperCumulativeEnergy,upperRightCumulativeEnergy;
    double minimumCumulativeEnergyUntilNow;

    Mat cumulativeEnergyMap = Mat(rowsize,colsize,CV_64F,double(0));

    energyImage.row(0).copyTo(cumulativeEnergyMap.row(0));

    for(int row = 1;row < rowsize;row++){
        for(int col = 0;col < colsize;col++){
            upperLeftCumulativeEnergy = cumulativeEnergyMap.at<double>(row - 1,max(col - 1,0)); //max function to handle the left most column(which doesn't have col - 1)
            upperCumulativeEnergy = cumulativeEnergyMap.at<double>(row-1,col);
            upperRightCumulativeEnergy = cumulativeEnergyMap.at<double>(row - 1,min(col + 1,colsize-1)); //min function to handle the right most column(which doesn't have col + 1)

            minimumCumulativeEnergyUntilNow = min(min(upperLeftCumulativeEnergy,upperCumulativeEnergy),upperRightCumulativeEnergy);

            cumulativeEnergyMap.at<double>(row,col) = energyImage.at<double>(row,col) + minimumCumulativeEnergyUntilNow;
        }
    }


    // create and show the newly created cumulative energy map converting map into color

    Mat color_cumulative_energy_map;
    double Cmin;
    double Cmax;
    minMaxLoc(cumulativeEnergyMap, &Cmin, &Cmax);
    float scale = 255.0 / (Cmax - Cmin);
    cumulativeEnergyMap.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
    applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET);
    namedWindow("Cumulative Energy Map", WINDOW_NORMAL); imshow("Cumulative Energy Map", color_cumulative_energy_map);waitKey(0);


//    namedWindow("Cumulative energy map",WINDOW_NORMAL);imshow("Cumulative energy map",cumulativeEnergyMap);

    return cumulativeEnergyMap;
}


int main(int argc, char** argv) {
    string imageName = "images/surfer.jpg";
    Mat image;
    image = imread(imageName,IMREAD_COLOR);
    Mat energyImage = getEnergyImage(image);
    Mat cumulativeEnergyImage = getCumulativeEnergyMap(energyImage);
    return 0;
}
