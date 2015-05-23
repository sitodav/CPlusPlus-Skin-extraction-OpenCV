
#pragma once
#include <opencv/cv.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "egbis.h"  //header made by christoffer holmestedt (https://github.com/christofferholmstedt/opencv-wrapper-egbisas) an opencv wrapper of Pedro Felzenswalb code for color segmentation

using namespace std;
using namespace cv;



Scalar calcolaDeviazioniStandard(Mat roi[], Scalar medie); //function used to compute the standard deviation in the B,G,R channels 
Scalar calcolaMedieMatSparsa(Mat img, vector<int> indici); //function used to compute the average intensities in the B,G,R channels, for a sparse mat
Scalar calcolaDevMatSparsa(Mat img,Scalar medie,vector<int> indici); //function used to compute the standard deviation in the BGR channels, for a sparse Mat
vector<int> eliminaOutliers(Mat img,vector<int> indici,Scalar medie, Scalar devStand);

void callBackEstrazioneTarget(int,int,int,int,void *); //routine used to interact with the window when we want to acquire the 6 skin class measures
void salvaValoriClassiTarget(char *path_img_classi, char * path_valori); //routine used to write the 6 measurements on file ( averages in the RGB channels, and standard dev in the RGB channels)
Scalar ** caricaValoriClassiDaFile(char * path); //routine used to read the 6 measurements from file

map<uchar,vector<int> > labelRegioni(Mat img); //function used to label the regions obtained with the Felzenswalb color segmentation routines
void stampaRegioneSparsa(Mat img,vector<int> indici); //used to display a sparse Mat

int classificaPelleImmagine(Mat img,Scalar **valori); //the routine that makes all the "dirty" work