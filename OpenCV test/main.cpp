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

    Sobel(image_grey,gradientX,ddepth,1,0);
    Sobel(image_grey,gradientY,ddepth,0,1);

//    Scharr(image_grey,gradientX, ddepth,1,0);
//    Scharr(image_grey,gradientY, ddepth,0,1);


    convertScaleAbs(gradientX,absGradientX);
    convertScaleAbs(gradientY,absGradientY);

    addWeighted(absGradientX,0.5,absGradientY,0.5,0,gradient);
    namedWindow("Gradient",WINDOW_NORMAL);
    imshow("Gradient",gradient);
    imwrite("images/gradient_bw.jpg",gradient);
    waitKey(0);
    return gradient;
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

            cumulativeEnergyMap.at<double>(row,col) = energyImage.at<double>(row,col);
        }
    }
    //namedWindow("Cumulative energy map",WINDOW_NORMAL);
    //imshow("Cumulative energy map",cumulativeEnergyMap);
    //imwrite("images/cumulative_energy_map.jpg",cumulativeEnergyMap);
    return cumulativeEnergyMap;
}


int main(int argc, char** argv) {
    string imageName = "images/scenery.jpg";
    Mat image;
    image = imread(imageName,IMREAD_COLOR);
    Mat energyImage = getEnergyImage(image);

    Mat cumulativeEnergyMap = getCumulativeEnergyMap(energyImage);
    return 0;
}
