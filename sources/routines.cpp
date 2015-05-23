#include "routines.h"

Mat imgClassi;
int xL,yL,xR,yR;
Mat * roi;
int step;


//callback function used to interact with a cv::Mat when we have to extract some ROIs (representing the 6 skin classes)
void callBackEstrazioneTarget(int event,int x,int y,int flags,void * params)
{
	//we just want to extract 6 ROIs
	if(step == 6)
	{
		destroyWindow("Seleziona");
		return;
	}
	switch(event)
	{
		case EVENT_LBUTTONDOWN:
			xL = x;
			yL = y;

		break;
		case EVENT_LBUTTONUP:
			xR = x;
			yR = y;

			int t;
			if(xL > xR){
				t = xL;
				xL = xR;
				xR = t;
			}
			if( yL > yR)
			{
				t = yL;
				yL = yR;
				yR = t;
			}

			//cout<<"xl : "<<xL<<" yl: "<<yL<<" xR: "<<xR<<" yR: "<<yR<<endl; 
			cout<<step<<endl;

			
			Mat blurred = imgClassi;

			GaussianBlur(imgClassi,blurred,Size(9,9),3);

			roi[step] = imgClassi(Rect(xL,yL,xR-xL,yR-yL));


			step++;
			 
	}
}


void salvaValoriClassiTarget(char *path_img_classi, char * path_valori)
{
	step = 0;
	roi = new Mat[6];
	
	imgClassi = imread(path_img_classi,CV_LOAD_IMAGE_COLOR);

	


	imshow("Seleziona",imgClassi);
	setMouseCallback("Seleziona",callBackEstrazioneTarget, NULL);
	waitKey(0);

	
	
	//now we have the 6 rois, representing the 6 skin classes
	//for each one of them we compute the averages and standard deviations in the 3 channels
	

	
	Scalar **valori = (Scalar **)new Scalar *[6];
	Mat channels[3];

	for(int i=0;i<6;i++)
	{

		valori[i] = (Scalar *) new Scalar[2]; //one scalar for the 3 (RGB) averages, and one for the 3 standard dev
		

		Scalar t = mean(roi[i]), t2; 
		
		for(int j=0;j<3;j++)
		{
			valori[i][0].val[j] = t[j];
		}
		//to compute standard dev we use our routine
		split(roi[i],channels);
		t2 = calcolaDeviazioniStandard(channels,t); //in t we still have the averages ...
		//ricopio in valori
		for (int j=0;j<3;j++)
		{
			valori[i][1].val[j] = t2[j];
		}
		
	}

	//debug printing

	for(int i=0;i<6;i++)
	{
		cout<<"per classe "<<i<<" red: "<<valori[i][0].val[2]<<" green: "<<valori[i][0].val[1]<<" blue: "<<valori[i][0].val[0]<<endl;
		cout<<"per classe "<<i<<" dev red: "<<valori[i][1].val[2]<<" dev green: "<<valori[i][1].val[1]<<" dev blue: "<<valori[i][1].val[0]<<endl<<"--------------------------------"<<endl;
	}

	//we show the 6 rois

	cout<<"CLASSI ACQUISITE"<<endl;
	char title[2];

	for(int i=0;i<6;i++)
	{
		sprintf(title,"%d",i);
		imshow(title,roi[i]);
		
	}

	waitKey(0);

	//we wrote them on file
	// mediar|mediag|mediab//devred|devgreen|devblue
	ofstream outputFile(path_valori);
	for(int i=0;i<6;i++)
	{
		outputFile<<valori[i][0].val[2]<<"|"<<valori[i][0].val[1]<<"|"<<valori[i][0].val[0]<<"?"<<valori[i][1].val[2]<<"|"<<valori[i][1].val[1]<<"|"<<valori[i][1].val[0]<<endl;
	}

	outputFile.close();

}


Scalar ** caricaValoriClassiDaFile(char * path)
{
	ifstream fileInput(path);
	string line, line2,token;
	int iClasse = 0;

	//we prepare the data structure
	Scalar **valori = (Scalar **)new Scalar *[6];
	for(int i=0;i<6;i++)
		valori[i] = new Scalar[2];


	while(getline(fileInput,line))
	{
		//in line we have a line from file
		//we make it a "stream"
		stringstream lineStr (line),lineStr2,lineStr3;
		//we extract the first part (before the ? char) which is about the averages
		getline(lineStr,line2,'?');
		//cout<<line2<<endl;
		lineStr2 = stringstream(line2);
		int k = 0;
		while(getline(lineStr2,token,'|'))
		{
			lineStr3 = stringstream(token);
			lineStr3 >> valori[iClasse][0].val[2-k];
			k++;
		}
		
		//we take the 3 numerical values and save them in the data structure used for the averages (restoring opencv bgr order, while on the file we store as rgb)
		

		//we extract the second part (the one for the standard devs)
		getline(lineStr,line2);
		
		lineStr2 = stringstream(line2);
		k = 0;
		while(getline(lineStr2,token,'|'))
		{
			lineStr3 = stringstream(token);
			lineStr3 >> valori[iClasse][1].val[2-k];
			k++;
		}

		iClasse++;
	}

	fileInput.close();

	//debug printing
	for(int i=0;i<6;i++)
	{
		cout<<"per classe "<<i<<" red: "<<valori[i][0].val[2]<<" green: "<<valori[i][0].val[1]<<" blue: "<<valori[i][0].val[0]<<endl;
		cout<<"per classe "<<i<<" dev red: "<<valori[i][1].val[2]<<" dev green: "<<valori[i][1].val[1]<<" dev blue: "<<valori[i][1].val[0]<<endl<<"--------------------------------"<<endl;
	}

	return valori;

}



Scalar calcolaDeviazioniStandard(Mat roi[], Scalar medie)
{

	Scalar *toReturn = new Scalar();

	
			for(int k=0;k<3;k++)
			{
				float somma = 0.0;

				for(int i=0;i<roi[k].rows;i++)
				{
					for(int j=0;j<roi[k].cols;j++)
					{
						somma += pow(( roi[k].at<unsigned char>(i,j) - medie.val[k] ), 2); 
					}
				}

				toReturn->val[k] = sqrt(somma / (roi[k].cols * roi[k].rows) );
			}


			return *toReturn;
	
	
}


Scalar calcolaMedieMatSparsa(Mat img, vector<int> indici)
{
		Scalar *toReturn = new Scalar();

		for(int k=0;k<3;k++)
		{
			float sum  = 0.0f;
			for(vector<int>::iterator it = indici.begin(); it!= indici.end(); it++)
			{
				int iR = (*it) / img.cols;
				int iC = (*it) % img.cols;
				sum += img.at<Vec3b>(iR,iC)[k];
			}
			toReturn->val[k] = sum;

		}


		toReturn->val[0] = (toReturn->val[0]) / indici.size();
		toReturn->val[1] = (toReturn->val[1]) / indici.size();
		toReturn->val[2] = (toReturn->val[2]) / indici.size();
		

		return *toReturn;
}

vector<int> eliminaOutliers(Mat img,vector<int> indici,Scalar medie, Scalar devStand)
{
	vector<int> *toReturn = new vector<int>();

	for(vector<int>::iterator it = indici.begin(); it!=indici.end(); it++)
	{
		int iR = (*it) / img.cols;
		int iC = (*it) % img.cols;

		uchar b = img.at<Vec3b>(iR,iC)[0];
		uchar g = img.at<Vec3b>(iR,iC)[1];
		uchar r = img.at<Vec3b>(iR,iC)[2];
	
		if( sqrt( pow(b-medie.val[0],2) + pow(g-medie.val[1],2) +  pow(r-medie.val[2],2)) < 2.0 * sqrt( pow(devStand.val[0],2) + pow(devStand.val[1],2) + pow(devStand.val[2],2) )  )
		{
			toReturn->push_back(*it);
		}
	
	}

	return *toReturn;
}

Scalar calcolaDevMatSparsa(Mat img, Scalar medie ,vector<int> indici)
{
		Scalar *toReturn = new Scalar();

		for(int k=0;k<3;k++)
		{
			float sum  = 0.0f;
			for(vector<int>::iterator it = indici.begin(); it!= indici.end(); it++)
			{
				int iR = (*it) / img.cols;
				int iC = (*it) % img.cols;
				sum += pow( img.at<Vec3b>(iR,iC)[k] - medie.val[k]  ,2);
			}
			toReturn->val[k] = sum;

		}

		toReturn->val[0] = sqrt((toReturn->val[0]) / indici.size());
		toReturn->val[1] = sqrt((toReturn->val[1]) / indici.size());
		toReturn->val[2] = sqrt((toReturn->val[2]) / indici.size());

		

		return *toReturn;


}




map<uchar,vector<int> > labelRegioni(Mat img)
{
	map<uchar,vector<int> > * mapToReturn = new map<uchar, vector<int> >(); 
	


	for(int i=0;i< img.rows; i++)
	{
		for(int j=0;j< img.cols; j++)
		{
			uchar colore = img.at<uchar>(i,j);
			if( mapToReturn->find(colore) == mapToReturn->end() )
			{
				(* mapToReturn)[colore] = vector<int>();
			}

			(* mapToReturn)[colore].push_back(i * img.cols + j);

		}
	}
	
	
	//per debug
	/*
	for(map<uchar,vector<int> >::iterator it = mapToReturn->begin(); it != mapToReturn->end(); it++)
	{
		Mat imgT(img.rows,img.cols,CV_8UC1);
		vector<int> indici = it->second;
		
		for(vector<int>::iterator it2 = indici.begin(); it2 != indici.end(); it2++)
		{
			int iR = (*it2) / img.cols;
			int iC = (*it2) % img.cols;
			imgT.at<uchar>(iR,iC) = 0;
			 
		}
		imshow("test debug",imgT);
		waitKey(0);
	}*/



	return * mapToReturn;
}


void stampaRegioneSparsa(Mat img,vector<int> indici)
{
	Mat imgT(img.rows,img.cols,CV_8UC3);
	for(vector<int>::iterator it2 = indici.begin(); it2 != indici.end(); it2++)
		{
			int iR = (*it2) / img.cols;
			int iC = (*it2) % img.cols;
			imgT.at<Vec3b>(iR,iC) = img.at<Vec3b>(iR,iC);
			 
		}
	imshow("test debug",imgT);
	waitKey(0);
}




int classificaPelleImmagine(Mat img,Scalar ** valori)
{
	Mat hsv;
	 
	//Felzenszwalb color segmentation algorithm works in 3 channels. We need it to work in HSH (hsv with h replacing the v channel). To do so, we convert
	//the original image in hsv, we split in the 3 hsv channels, and we put h in place of v channel. In this way, the color segmentation will give much more "importance" to the hue channel

	//so i convert the original image in hsv
	cvtColor(img,hsv,CV_RGB2HSV);
	imshow("hsv",hsv); waitKey(0);
	//i split in 3 channels
	Mat channels[3];
	split(hsv,channels);
	
	//I replace V channel with H channel
	channels[2] = channels[0].clone();
	
	Mat hue3c;
	//and we "remerge"
	merge(channels,3,hue3c);
	 
	

	//we run the wrapper for Felzenszwalb segmentation routine
	int n;
	Mat imgSegmentata = runEgbisOnMat(hue3c, 0.5, 500.0f, 20, &n);
	//and we show the result of the segmentation
	imshow("seg",imgSegmentata); waitKey(0);


	//we convert the segmented image in grayscale, using the gray levels as labels (TO-DO: this approach it's not robust !! Fix this)
	Mat imgSegmentataGS;
	cvtColor(imgSegmentata,imgSegmentataGS,CV_BGR2GRAY);
	Mat imgBlurred;




	//end of segmentation ------------------------------------------------------------------------------------------

	
	//classifichiamo i pixel dell'immagine segmentata, per ottenere le regioni connesse
	map<uchar,vector<int> > mappaRegioni = labelRegioni(imgSegmentataGS);
	
	map<uchar, Scalar> medie, devStand;
	
	

	GaussianBlur(img,imgBlurred,Size(23,23),23);

	
	
	//i compute the averages for the labeled regions 
	for(map<uchar,vector<int> >::iterator it = mappaRegioni.begin(); it!= mappaRegioni.end(); it++)
	{
		medie[ it->first ] = calcolaMedieMatSparsa(img, it->second );
	}


	//and the standard devs
	for(map<uchar, vector<int> >::iterator it = mappaRegioni.begin(); it!=mappaRegioni.end(); it++)
	{
		devStand [ it-> first ] = calcolaDevMatSparsa(img, medie[it->first], it->second );
	}

	//we remove the outliers
	for(map<uchar, vector<int> >::iterator it = mappaRegioni.begin(); it!=mappaRegioni.end(); it++)
	{
		mappaRegioni[it->first] = eliminaOutliers(img,mappaRegioni[it->first],medie[it->first],devStand[it->first]);
	}
	


	//for every region i compute the differences between their 3 averages (RGB) and the 3 averages of EVERY skin class

	map<uchar , Scalar> differenze;
	map<uchar, int> indiciClassiScelte;
	map<uchar, float> minDiffRegioneClasse;

	
	//I want to keep the nearest SKIN class for every region
	for(map<uchar,vector<int> >::iterator it = mappaRegioni.begin(); it!= mappaRegioni.end(); it++)
	{
		uchar chiaveRegione = it-> first;
		for(int i = 0;i<6;i++) //for each and every one of the 6 skin classes
		{
			//we leave the abs here for clarity (and for testing) but it's not needed because the values are all positives, and we'll use euclidean distance
			Scalar diff ( abs( medie[chiaveRegione].val[0] - valori[i][0].val[0] ) , abs( medie[chiaveRegione].val[1] - valori[i][0].val[1] ) , abs( medie[chiaveRegione].val[2] - valori[i][0].val[2])) ;
			
			//cout<<diff.val[0]<< " " <<diff.val[1] << " " <<diff.val[2]<<endl;

			//We compare the total difference (between the rgb averages of the region, and the rgb averages of the skin class) as euclidean distances, with the previous measurements
			//from the other classes
			if(i == 0)
			{
				//i always add at the first iteration
				indiciClassiScelte [chiaveRegione] = i;
				differenze[ chiaveRegione ] = diff;
				minDiffRegioneClasse [chiaveRegione ] = sqrt(pow(diff.val[0],2) + pow(diff.val[1],2) + pow(diff.val[2],2) );

				 
			}
			else 
			{
				if( sqrt( pow(differenze[chiaveRegione].val[0],2) + pow(differenze[chiaveRegione].val[1],2) + pow(differenze[chiaveRegione].val[2],2) ) > sqrt(pow(diff.val[0],2) + pow(diff.val[1],2) + pow(diff.val[2],2) ) )
				{//i substitute because for every region i will keep only the NEAREST skin class
					differenze[chiaveRegione] = diff;
					indiciClassiScelte[chiaveRegione] = i ;
					minDiffRegioneClasse[chiaveRegione] = sqrt(pow(diff.val[0],2) + pow(diff.val[1],2) + pow(diff.val[2],2) );
				}
				
			}
		}
		
	}


	
	//i need to store the maximum distance of a region from its skin class
	float MAXDIFF = 0.0f;
	
	for(map<uchar,float>:: iterator it = minDiffRegioneClasse.begin(); it != minDiffRegioneClasse.end(); it ++ )
	{
		if ( it->second > MAXDIFF )
		{
			MAXDIFF = it->second;
		}
	}
	//and we'll use that distance to normalize the distance of each region (from its skin class) between 1 and 0 (1 for the regions which are the nearest to their skin classes)
	

	//we compute the score of a region
	//the score is computed using the normalized distance of the region, and the size of the region
	map<uchar, float> punteggi;
	uchar chiaveWin;
	float maxScore = 0.0f;
	for(map<uchar,vector<int> >::iterator it = mappaRegioni.begin(); it!= mappaRegioni.end(); it++)
	{
		float puntDiff = abs((MAXDIFF-minDiffRegioneClasse[it->first])) / MAXDIFF;
		//float puntSize = it->second.size() / (img.rows * img.cols);

		//the region score is the normalized distance from the skin class multiplied the size of the region
		//we only keep the region with the highest score
		punteggi[it->first] = puntDiff * it->second.size();
		if(punteggi[it->first] > maxScore)
		{
			maxScore = punteggi[it->first];
			chiaveWin = it-> first;
		}

	}


	
	//we display the region with the highest score
	stampaRegioneSparsa(img,mappaRegioni[chiaveWin]);

	return indiciClassiScelte[chiaveWin];
	





}


