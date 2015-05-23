The code written for the Homework is contained in the Source (.cpp) file, and in routines (.cpp and .h) files.
The application uses an opencv wrapper (made by Christoffer Holmstedt, and available at https://github.com/christofferholmstedt/opencv-wrapper-egbis) of the
segmentation algorithm based on graph partitioning made by Pedro Felzenszwalb and available at http://cs.brown.edu/~pff/.
The code for the wrapper is contained in the "seg" folder.
The entry point for the application is in the Source.cpp folder. It depends on "routines.h" (which depends on egbis.h, contained in seg/opencv-wrapper-egbis-master).
All the files were imported in Visual Studio 2012 in a project configured to work with OpenCV 3.0.
So to compile the program, just copy all the sources and headers in the same project (configured to use opencv), change the 2 #defines (PATH_IMMAGINE_CLASSI_TARGET & PATH_FILE_VALORI_CLASSI)
giving the paths to the 6 skin classes image (FitzpatrickFaces.png) and the path of the output measurements file, and it's ready to go (comment/uncomment the first line in the main
to use the cam or to read from file).
