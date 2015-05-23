//In this application we take some measurements from 6 labeled skin classes (using the salvaValoriClassiTarget routine) and analyse their averages values in RGB space (and standard deviation)
//We use Christoffer Holmestedt Opencv wrapper for "Efficient Graph-Based Image Segmentation algorithm" developed by Pedro F. Felzenszwalb and Daniel P. Huttenlocher
//to segment the image in the HSH SPACE (we convert the image in the HSV space, and we substitute the V channel with the H one....in this way Felzenszwalb routine, which works in 3 channels, seems
//to be more robust)

//both the wrapper (and the original code from Felzenszwalb) are available at:

//https://github.com/christofferholmstedt/opencv-wrapper-egbis for the Holmestedt wrapper
//http://cs.brown.edu/~pff/ for Felzenszwalb original segmentation algorithm

#include <iostream>
#include "routines.h"
#include <time.h>


//change this
#define PATH_IMMAGINE_CLASSI_TARGET "C:/Users/davide/Desktop/progetto_embedded/FitzpatrickFaces.png" //this is the path where we keep the image used to acquire the 6 skin classes
#define PATH_FILE_VALORI_CLASSI "C:/Users/davide/Desktop/progetto_embedded/valori.txt" //this is where we save the 6 skin classes measurements 

int main()
{
	srand((unsigned int)time(NULL));

	//uncomment this next line if you want to acquire a new "set" of measurements for the 6 skin classes
	//salvaValoriClassiTarget(PATH_IMMAGINE_CLASSI_TARGET, PATH_FILE_VALORI_CLASSI);

	//we read the 6 measurement (previously acquired) of skin classes, from file
	Scalar ** valori = caricaValoriClassiDaFile(PATH_FILE_VALORI_CLASSI);


	//uncomment this line if you want to test the program using an image from disk
	/*Mat img = imread("C:/Users/davide/Desktop/progetto_embedded/test.jpg",CV_LOAD_IMAGE_COLOR);
	
	cout<<"skin color class is "<<classificaPelleImmagine(img,valori)<<endl;
	getchar();*/

	//here we capture from laptop camera
	
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		cout<<"errore apertura cam"<<endl;
		return -1;
	}

	namedWindow("Frame da cam",1);

	while(1)
	{

		Mat nuovoFrame;
		cap >> nuovoFrame;
		imshow("Frame da cam",nuovoFrame);
		char c;
		//when we play 'f' the frame is captured and "classified"
		if((c=waitKey(30)) == 'f')
		{
				int skinClass = classificaPelleImmagine(nuovoFrame,valori);
				cout<<"the skin class for the frame is : "<<skinClass<<endl;
		}
	}
	


}
