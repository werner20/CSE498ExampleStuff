#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "FaceTracker.h"

using namespace cv;
using namespace std;

void BlurArea( Mat& image, const Rect& area );

Mat image;
bool selectObject = false;
int trackObject = 0;
Point origin;
Rect selection;

static void onMouse( int event, int x, int y, int, void* )
{
   if ( selectObject )
   {
      selection.x = MIN( x, origin.x );
      selection.y = MIN( y, origin.y );
      selection.width = std::abs( x - origin.x );
      selection.height = std::abs( y - origin.y );

      selection &= Rect( 0, 0, image.cols, image.rows );
   }

   switch ( event )
   {
   case EVENT_LBUTTONDOWN:
      origin = Point( x, y );
      selection = Rect( x, y, 0, 0 );
      selectObject = true;
      break;

   case EVENT_LBUTTONUP:
      selectObject = false;

      if ( selection.width > 0 && selection.height > 0 )
      {
         trackObject = -1;
      }

      break;
   }
}

int main( int argc, const char** argv )
{
   if ( argc < 3 || argc > 4 )
   {
      cerr << "Incorrect number of arguments. The following is expected:\n\n"
         << "First:  Video file to be processed.\n"
         << "Second: The name of the output video.\n"
         << "Third: Optional. Frame from which to start reading.\n\n";

      //return -1;
   }

   string inputFile( "Test1.mp4" );
   string outputFile( "Test1Output2" );
   auto resizeFactor = 1.0;

   auto startFrame = 1;
   if ( argc == 4 )
   {
      startFrame = stoi( argv[ 3 ] );
   }

   auto outputVideo = outputFile + string( ".avi" );

   
   VideoCapture cap;
   VideoWriter videoWriter;
   Mat frame;

   cap.open( inputFile );
   if ( !cap.isOpened() )
   {
      cout << "Failed to open file: " + inputFile << endl << endl;
      //return -1;
   }

   /*Named window creats a window that can be used a placeholder. Created windows are referred to by their name*/
   namedWindow( "BlurredFaces", 0 );
   setMouseCallback( "BlurredFaces", onMouse, nullptr );

   for ( auto i = 0; i < startFrame; i++ )
   {
      cap >> frame;
   }

   if ( resizeFactor != 1.0 )
   {
      resize( frame, frame, Size( static_cast< int >( frame.cols * resizeFactor ), static_cast< int >( frame.rows * resizeFactor ) ) );
   }

   FaceTracker faceTracker;
   Rect objectLocation;

   auto paused = true;
   for ( ;; )
   {
      if ( !paused )
      {
         cap >> frame;
         if ( frame.empty() )
         {
            videoWriter.release();
            break;
         }
      }

      if ( image.empty() )
      {
         frame.copyTo( image );
      }

      if ( !paused )
      {
         if ( resizeFactor != 1.0 )
         {
            resize( frame, frame, Size( static_cast< int >( frame.cols * resizeFactor ), static_cast< int >( frame.rows * resizeFactor ) ) );
         }

         if ( trackObject )
         {
            if ( trackObject < 0 )
            {
               videoWriter.open( outputVideo, CV_FOURCC( 'X', 'V', 'I', 'D' ), 24, Size( frame.cols, frame.rows ) );
               if ( !videoWriter.isOpened() )
               {
                  cout << "Failed to open output video." << endl;
               }

               trackObject = 1;
               faceTracker.Initialize( frame, selection );
            }
            else
            {
               faceTracker.TrackNextFrame( frame, objectLocation );
            }
         }
      }
      else if ( trackObject < 0 )
      {
         paused = false;
      }

      if ( videoWriter.isOpened() )
      {
         auto blurredObject( frame );
         BlurArea( blurredObject, objectLocation );

         videoWriter << blurredObject;

         imshow( "BlurredFaces", blurredObject );
         waitKey( 1 );
      }
      else
      {
         putText( image, "Select region", Point( 20, 50 ), FONT_HERSHEY_PLAIN, 1.5, Scalar( 0, 0, 255, 255 ), 2 );

         imshow( "BlurredFaces", image );
         waitKey( 1 );
      }
   }

   std::cout << "Enter a number to continue: ";
   int num;
   std::cin >> num;
   return 0;
}

void BlurArea( Mat& image, const Rect& area )
{
   auto roi = image( area );
   GaussianBlur( roi, roi, Size( 23, 23 ), 30 );
}