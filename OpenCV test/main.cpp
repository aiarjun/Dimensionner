#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

enum seamDirection { VERTICAL, HORIZONTAL };
seamDirection SeamDirection;

bool demo;


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
        namedWindow("Energy Map", WINDOW_NORMAL); imshow("Energy Map", energyImage);waitKey(0);
    }
    return energyImage;
}


Mat getCumulativeEnergyMap(Mat& energyImage){
    int rowsize = energyImage.rows;
    int colsize = energyImage.cols;

    double upperLeftCumulativeEnergy,upperCumulativeEnergy,upperRightCumulativeEnergy;
    double minimumCumulativeEnergyUntilNow;

    Mat cumulativeEnergyMap = Mat(rowsize,colsize,CV_64F,double(0));

    if (SeamDirection == VERTICAL) energyImage.row(0).copyTo(cumulativeEnergyMap.row(0));
    else if (SeamDirection == HORIZONTAL) energyImage.col(0).copyTo(cumulativeEnergyMap.col(0));


    if (SeamDirection == VERTICAL){
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
    else if(SeamDirection==HORIZONTAL){
            for(int col = 1;col < colsize;col++){
            for(int row = 0;row < rowsize;row++){
                upperLeftCumulativeEnergy = cumulativeEnergyMap.at<double>(max(row - 1, 0), col - 1); //max function to handle the left most column(which doesn't have col - 1)
                upperCumulativeEnergy = cumulativeEnergyMap.at<double>(row, col - 1);
                upperRightCumulativeEnergy = cumulativeEnergyMap.at<double>(min(row + 1, rowsize - 1), col - 1); //min function to handle the right most column(which doesn't have col + 1)

                minimumCumulativeEnergyUntilNow = min(min(upperLeftCumulativeEnergy,upperCumulativeEnergy),upperRightCumulativeEnergy);

                cumulativeEnergyMap.at<double>(row,col) = energyImage.at<double>(row,col) + minimumCumulativeEnergyUntilNow;
            }
        }

    }


     //create and show the newly created cumulative energy map converting map into color

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

    if(SeamDirection==VERTICAL){
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
    }

    else if(SeamDirection==HORIZONTAL){
        Mat lastCol = cumulativeEnergyMap.col(colsize - 1);
        minMaxLoc(lastCol,&minVal,&maxVal,&minPoint,&maxPoint);

        int minRowIndex = minPoint.y;

        path.resize(colsize);
        path[colsize - 1] = minRowIndex;

        for(int col = colsize - 2;col >= 0;col--){
            a = cumulativeEnergyMap.at<double>(max(minRowIndex - 1, 0), col);
            b = cumulativeEnergyMap.at<double>(minRowIndex, col);
            c = cumulativeEnergyMap.at<double>(min(minRowIndex + 1, rowsize - 1), col);

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

            minRowIndex += offset;
            //minRowIndex = min(max(minRowIndex, 0), colsize - 1);
            if(minRowIndex > rowsize - 1){
                minRowIndex = rowsize - 1;
            }
            else if(minRowIndex < 0){
                minRowIndex = 0;
            }

            path[col] = minRowIndex;
        }
    }

    //we have now obtained a path from the last row to the first row, storing all the column values

    //now we have to remove that seam

    return path;

}


void showPath(Mat energyImage,vector<int> path){

    if(SeamDirection==VERTICAL){
        for(int row = 0;row < energyImage.rows;row++){
            energyImage.at<double>(row,path[row]) = 1;
        }
    }
    else if(SeamDirection==HORIZONTAL){
        for(int col = 0;col < energyImage.cols;col++){
            energyImage.at<double>(path[col],col) = 1;
        }
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

    if(SeamDirection==VERTICAL){
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
    }

    if(SeamDirection==HORIZONTAL){
           for(int col = 0;col < colsize;col++){
            Mat requiredCol;
            Mat lowerPartOfCol = image.colRange(col,col + 1).rowRange(0,path[col]);
            Mat upperPartOfCol = image.colRange(col,col + 1).rowRange(path[col] + 1,rowsize);

            if(!lowerPartOfCol.empty() && !upperPartOfCol.empty()){
                vconcat(lowerPartOfCol,upperPartOfCol,requiredCol);
                vconcat(requiredCol,dummyColumn,requiredCol);
            }
            else if(lowerPartOfCol.empty()){
                vconcat(upperPartOfCol,dummyColumn,requiredCol);
            }
            else{
                vconcat(lowerPartOfCol,dummyColumn,requiredCol);
            }

            requiredCol.copyTo(image.col(col));
        }

        image = image.rowRange(0,rowsize - 1);

    }
    if(demo){
        namedWindow("image",WINDOW_NORMAL);imshow("Reduced image",image);waitKey(0);
    }

    return image;
}

int main(int argc, char** argv) {

    demo = false;
    string outputname = "images/outputs/";
    string imageName;
    cout<<"Image path : ";
    cin>>imageName;
    Mat image = imread(imageName,IMREAD_COLOR);
    if(image.empty()){
        exit(1);
    }

    int viterations;
    cout<<"Vertical Iterations : ";
    cin>>viterations;

    int hiterations;
    cout<<"Horizontal Iterations : ";
    cin>>hiterations;


    SeamDirection = HORIZONTAL;
    for(int i = 0;i < hiterations;i++){
        cout<<i<<endl;
        Mat energyImage = getEnergyImage(image);
        Mat cumulativeEnergyImage = getCumulativeEnergyMap(energyImage);
        vector<int> path = findOptimalSeam(cumulativeEnergyImage);
        showPath(energyImage,path);
        Mat reducedImage = reduce(image,path);
        imwrite(outputname+to_string(i)+".jpg",reducedImage);
    }

    SeamDirection = VERTICAL;
    for(int i = 0;i < viterations;i++){
        cout<<i<<endl;
        Mat energyImage = getEnergyImage(image);
        Mat cumulativeEnergyImage = getCumulativeEnergyMap(energyImage);
        vector<int> path = findOptimalSeam(cumulativeEnergyImage);
        showPath(energyImage,path);
        Mat reducedImage = reduce(image,path);
    }

    Mat energyImage = getEnergyImage(image);
    Mat cumulativeEnergyImage = getCumulativeEnergyMap(energyImage);
    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", energyImage );
    imwrite("images/reducedimage.jpg",image);



    return 0;
}
