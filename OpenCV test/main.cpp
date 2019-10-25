#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

bool useForwardEnergy = true;

bool demo = false;

double squareDifference(double x,double y){
    return pow((x-y),2);
}

double getColorDifference(Mat image,double row1,double col1,double row2,double col2){
    double blue1 = image.at<double>(row1,col1)[0];
    double green1 = image.at<double>(row1,col1)[1];
    double red1 = image.at<double>(row1,col1)[2];
    double blue2 = image.at<double>(row2,col2)[0];
    double green2 = image.at<double>(row2,col2)[1];
    double red2 = image.at<double>(row2,col2)[2];

    double difference = squareDifference(blue1,blue2) + squareDifference(green1,green2) + squareDifference(red1,red2);

    return difference;
}

vector<int> getForwardCosts(Mat image,double row,double col){

    double l = 0,u = 0,r = 0;

    int rowsize = energyImage.rows;
    int colsize = energyImage.cols;

    if(row == 0){
        u = getColorDifference(image,row,col-1,row,col+1);
    }
    else if(col == 0){
        u = getColorDifference(image,row,col,row,col+1);
        l = getColorDifference(image,row,col,row,col+1) + getColorDifference(image,row,col,row-1,col+1);
        r = getColorDifference(image,row,col,row,col+1) + getColorDifference(image,row,col+1,row-1,col);
    }
    else if(col == colsize - 1){
        u = getColorDifference(image,row,col,row,col-1);
        l = getColorDifference(image,row,col,row,col-1) + getColorDifference(image,row,col-1,row-1,col);
        r = getColorDifference(image,row,col,row,col-1) + getColorDifference(image,row,col,row-1,col);
    }
    else{
        u = getColorDifference(image,row,col-1,row,col+1);
        l = getColorDifference(image,row,col-1,row,col+1) + getColorDifference(image,row,col-1,row-1,col);
        r = getColorDifference(image,row,col-1,row,col+1) + getColorDifference(image,row,col+1,row-1,col);
    }
    vector<int> costs;

    costs.push_back(l);
    costs.push_back(u);
    costs.push_back(r);

    return costs;
}

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

//    Sobel(image_grey,gradientX,ddepth,1,0);
//    Sobel(image_grey,gradientY,ddepth,0,1);

    Scharr(image_grey,gradientX, ddepth,1,0);
    Scharr(image_grey,gradientY, ddepth,0,1);


    convertScaleAbs(gradientX,absGradientX);
    convertScaleAbs(gradientY,absGradientY);

    addWeighted(absGradientX,0.5,absGradientY,0.5,0,gradient);

    gradient.convertTo(energyImage,CV_64F,1.0/255.0);


    if(demo){
        namedWindow("Energy image",WINDOW_NORMAL);imshow("Energy image",energyImage);waitKey(0);
    }

    return energyImage;
}


Mat getCumulativeEnergyMap(Mat& energyImage){
    int rowsize = energyImage.rows;
    int colsize = energyImage.cols;

    double upperLeftCumulativeEnergy,upperCumulativeEnergy,upperRightCumulativeEnergy;
    double minimumCumulativeEnergyUntilNow;

    Mat cumulativeEnergyMap = Mat(rowsize,colsize,CV_64F,double(0));


    if(useForwardEnergy){
        for(int row = 0;row < rowsize;row++){
            for(int col = 0;col < colsize;col++){
                upperLeftCumulativeEnergy = cumulativeEnergyMap.at<double>(row - 1,max(col - 1,0)); //max function to handle the left most column(which doesn't have col - 1)
                upperCumulativeEnergy = cumulativeEnergyMap.at<double>(row-1,col);
                upperRightCumulativeEnergy = cumulativeEnergyMap.at<double>(row - 1,min(col + 1,colsize-1)); //min function to handle the right most column(which doesn't have col + 1)

                minimumCumulativeEnergyUntilNow = min(min(upperLeftCumulativeEnergy,upperCumulativeEnergy),upperRightCumulativeEnergy);

                cumulativeEnergyMap.at<double>(row,col) = energyImage.at<double>(row,col) + minimumCumulativeEnergyUntilNow;
            }
        }
    }
    else{
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
    }





    // create and show the newly created cumulative energy map converting map into color

    if(demo){
        Mat color_cumulative_energy_map;
        double Cmin;
        double Cmax;
        minMaxLoc(cumulativeEnergyMap, &Cmin, &Cmax);
        float scale = 255.0 / (Cmax - Cmin);
        cumulativeEnergyMap.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
        applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, cv::COLORMAP_JET);
        namedWindow("Cumulative Energy Map", WINDOW_NORMAL); imshow("Cumulative Energy Map", color_cumulative_energy_map);waitKey(0);
    }

    return cumulativeEnergyMap;
}


vector<int> findOptimalSeam(Mat& cumulativeEnergyMap){
    vector<int> path;
    double a,b,c;
    int rowsize = cumulativeEnergyMap.rows;
    int colsize = cumulativeEnergyMap.cols;
    double minVal,maxVal;
    Point minPoint,maxPoint;
    int offset;


    Mat lastRow = cumulativeEnergyMap.row(rowsize - 1);

    minMaxLoc(lastRow,&minVal,&maxVal,&minPoint,&maxPoint);

    int minColIndex = minPoint.x;

    path.resize(rowsize);
    path[rowsize - 1] = minColIndex;

    for(int row = rowsize - 2;row >= 0;row--){
        a = cumulativeEnergyMap.at<double>(row,max(minColIndex - 1,0));
        b = cumulativeEnergyMap.at<double>(row,minColIndex);
        c = cumulativeEnergyMap.at<double>(row,min(minColIndex + 1,colsize - 1));

        if(a < min(b,c)){
            //a is the least value
            offset = -1;
        }
        else if(b < min(a,c)){
            offset = 0;
        }
        else{
            offset = 1;
        }

        minColIndex += offset;
        if(minColIndex > colsize - 1){
            minColIndex = colsize - 1;
        }
        else if(minColIndex < 0){
            minColIndex = 0;
        }
        path[row] = minColIndex;
    }


    //we have now obtained a path from the last row to the first row, storing all the column values

    //now we have to remove that seam

    return path;

}


void showPath(Mat energyImage,vector<int> path){
    for(int row = 0;row < energyImage.rows;row++){
        energyImage.at<double>(row,path[row]) = 1;
    }

    if(demo){
        namedWindow("Optimal seam",WINDOW_NORMAL);imshow("Optimal seam",energyImage);waitKey(0);
    }
}

Mat reduce(Mat& image,vector<int> path){
    //copy all the rows to the new image

    int rowsize = image.rows;
    int colsize = image.cols;

    Mat dummyColumn(1,1,image.type(),Vec3b(0,0,0));


    for(int row = 0;row < rowsize;row++){
        Mat requiredRow;
        Mat leftPartOfRow = image.rowRange(row,row + 1).colRange(0,path[row]);
        Mat rightPartOfRow = image.rowRange(row,row + 1).colRange(path[row] + 1,colsize);

        if(!leftPartOfRow.empty() && !rightPartOfRow.empty()){
            hconcat(leftPartOfRow,rightPartOfRow,requiredRow);
            hconcat(requiredRow,dummyColumn,requiredRow);
        }
        else if(leftPartOfRow.empty()){
            hconcat(rightPartOfRow,dummyColumn,requiredRow);
        }
        else{
            hconcat(leftPartOfRow,dummyColumn,requiredRow);
        }

        requiredRow.copyTo(image.row(row));
    }

    image = image.colRange(0,colsize - 1);
    if(demo){
        namedWindow("Reduced image",WINDOW_NORMAL);imshow("Reduced image",image);waitKey(0);
    }
    return image;
}

int main(int argc, char** argv) {
    string imageName;
    cout<<"Image path : ";
    cin>>imageName;
    Mat image = imread(imageName,IMREAD_COLOR);
    if(image.empty()){
        exit(1);
    }
    int iterations;
    cout<<"Iterations : ";
    cin>>iterations;
    demo = false;
    cout<<"Demo ? <0/1> (Only one iteration) ";
    cin>>demo;
    if(demo){
        iterations = 1;
    }
    for(int i = 0;i < iterations;i++){
        cout<<i<<endl;
        Mat energyImage = getEnergyImage(image);
        Mat cumulativeEnergyImage = getCumulativeEnergyMap(energyImage);
        vector<int> path = findOptimalSeam(cumulativeEnergyImage);
        showPath(energyImage,path);
        Mat reducedImage = reduce(image,path); //as image is passed by reference, the variable reduced image is discarded
    }
    imwrite("images/reduced_image.jpg",image);
    return 0;
}
