
#include "FaceTracker.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <limits>
#include <cmath>

using namespace cv;
using namespace std;

FaceTracker::FaceTracker()
   : _kalmanFilter( 4, 2, 0 )
{
   _frontalCascade.load( "haarcascade_frontalface_alt2.xml" );
   _profileCascade.load( "haarcascade_profileface.xml" );
}

void FaceTracker::Initialize( const cv::Mat& firstFrame, cv::Rect objectBoundary )
{
   _trackWindow = objectBoundary;

   int initialX = _trackWindow.x + _trackWindow.width / 2;
   int initialY = _trackWindow.y + _trackWindow.height / 2;

   _kalmanFilter.statePre.at<float>( 0 ) = initialX;
   _kalmanFilter.statePre.at<float>( 1 ) = initialY;
   _kalmanFilter.statePre.at<float>( 2 ) = 0;
   _kalmanFilter.statePre.at<float>( 3 ) = 0;
   _kalmanFilter.transitionMatrix = *( Mat_<float>( 4, 4 ) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 );
   setIdentity( _kalmanFilter.measurementMatrix );
   setIdentity( _kalmanFilter.processNoiseCov, Scalar::all( 5 ) );
   setIdentity( _kalmanFilter.measurementNoiseCov, Scalar::all( 1 ) );
   setIdentity( _kalmanFilter.errorCovPost, Scalar::all( .1 ) );

   UpdateKalmanFilterAndTrackWindow( initialX, initialY );
}

bool FaceTracker::TrackNextFrame( const cv::Mat& frame, cv::Rect& objectBoundary )
{
	/*.predict "computes a predicted state". My understanding is that it predicts a location for the blur box
		before the next frame is actually processed and the predicted values are passed into the centering function
		for the blur box. Then the frame is actually processed and the blur box position is adjusted accordingly*/
   auto prediction = _kalmanFilter.predict();
   CenterTrackWindowAboutPosition( static_cast< int >( prediction.at<float>( 0 ) ), static_cast< int >( prediction.at<float>( 1 ) ) );

   Mat frameGray;
   cvtColor( frame, frameGray, CV_BGR2GRAY );
   equalizeHist( frameGray, frameGray );

   vector<Rect> faces;
   //This does the actual tracking? Takes in a a grayscale video frame and returns
   //detected objects as rects in faces
   _frontalCascade.detectMultiScale( frameGray, faces );

   vector<Rect> profileFaces;
   _profileCascade.detectMultiScale( frameGray, profileFaces );

   //make one vector of face rects including front and profile views
   faces.insert( faces.end(), profileFaces.begin(), profileFaces.end() );

   //the "maximum finite value". Of what? a double?
   double closestDistance = numeric_limits<double>::max();
    
   Rect closestFace;
   //iterate through all the faces and determine which face in the next frame 
   //is closes to the current one. logically this should be the same face, so 
   //move the blur?
   for ( auto face : faces )
   {
	   /*!!!!!!! CODE ADDED BY ALYSSA WERNER !!!!!

	   cv::Scalar colorScalar = cv::Scalar(94, 206, 165);
	   //cv::Point center(image.size().width*0.5, image.size().height*0.5);
	  // cv::Size size(100, 100);
	   //cv::ellipse(image, center, size, 0, 0, 360, colorScalar, 4, 8, 0);
	   cv::Point topLeft(face.x, face.y);
	   cv::Point lowerRight(face.x + face.width, face.y + face.height);
	   auto img = frame;

	   //format: C++: void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
	   cv::rectangle(frame, topLeft, lowerRight, colorScalar, 1, 8, 0);
		   //img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		   //roi_gray = gray[y:y + h, x : x + w]
		   //roi_color = img[y:y + h, x : x + w] }*/



      std::pair<int, int> faceCenter;

	  //the center x,y location of the face
      faceCenter.first = face.x + face.width / 2;
      faceCenter.second = face.y + face.height / 2;

	  //_trackWindow is set to the boundaries of the blur rectangle (x,y,w,h) where x and y are upper left corner
	  /*if the x location of the face center is less (further left) than the x/y boundary of the blur box or greater
		(further right) than it's right/bottom most x/y position continue
		
		I think the logic of this mechanism is meant to distinguish between faces? If it picks up a face outside 
		the blur window in the next frame, it's probably not the same face because it moved outside the range of
		the blur very quickly. This is probably why it lost track of Grant in example1 (ask me for clarification 
		if you need)>. If it is continue to the next face because we don't care about it*/
      if ( faceCenter.first < _trackWindow.x || faceCenter.first > _trackWindow.x + _trackWindow.width ||
         faceCenter.second < _trackWindow.y || faceCenter.second > _trackWindow.y + _trackWindow.height )
      {
         continue;
      }
	  //The calculate distance function calculates the distance between the center points of the blur box and the
	  //face rect
      double distance = CalculateDistance( _trackWindow, face );

	  //if the distance between them is less than the closest distance (still a bit unclear here?) set the 
	  //closest distance to distance and closest face to face
      if ( distance < closestDistance )
      {
         closestDistance = distance;
         closestFace = face;
      }
   }
   //again with the numeric_limits. WHAT ARE YOU???
   //	Update: it's a constant, apparently = 1.79769e+308

   /*Assuming it's the max numeric limit of a double that's really freaking huge so closest distance would 
	 always be less than it?*/
   if ( closestDistance < numeric_limits<double>::max() )
   {
	   //pass the center points of the closest face into the Filter updater
      UpdateKalmanFilterAndTrackWindow( closestFace.x + closestFace.width / 2, closestFace.y + closestFace.height / 2 );
   }
   /*if the x/y position is less than zero (how would this even happen???? anything less than 0 is off screen...maybe
   that's the point?) or if the width of the blur box is larger than the number of columns/rows in frame (which is a Mat
   object of the image) return false. So I think this is saying if the blur box is bigger than the actual frame image return
   false
   
   // frame.cols or frame.rows = the number of rows and columns or (-1, -1) when the array has more than 2 dimensions*/
   if ( _trackWindow.x < 0 || _trackWindow.y < 0 || _trackWindow.x + _trackWindow.width > frame.cols || _trackWindow.y + _trackWindow.height > frame.rows )
   {
      return false;
   }

   //if the blur box is a valid size set the objectBoundary to that and return true. If true is returned we've successfully
   //blurred the face in the next frame?
   objectBoundary = _trackWindow;

   return true;
}

void FaceTracker::CenterTrackWindowAboutPosition( int x, int y )
{
   _trackWindow.x = x - _trackWindow.width / 2;
   _trackWindow.y = y - _trackWindow.height / 2;
}

double FaceTracker::CalculateDistance( const cv::Rect& rect1, const cv::Rect& rect2 )
{
   std::pair<int, int> rect1Center;
   rect1Center.first = rect1.x + rect1.width / 2;
   rect1Center.second = rect1.y + rect1.height / 2;

   std::pair<int, int> rect2Center;
   rect2Center.first = rect2.x + rect2.width / 2;
   rect2Center.second = rect2.y + rect2.height / 2;

   return sqrt( ( rect2.x - rect1.x ) * ( rect2.x - rect1.x ) + ( rect2.y - rect1.y ) * ( rect2.y - rect1.y ) );
}
//pass in the center x and y positions
void FaceTracker::UpdateKalmanFilterAndTrackWindow( int observationCenterX, int observationCenterY )
{
	//I can't seem to find documentation on this so I'm not sure what measurement does, honestly
   Mat_<float> measurement( 2, 1 );
   measurement( 0 ) = observationCenterX;
   measurement( 1 ) = observationCenterY;

   //.correct updates the predicted state from the measurements
   auto estimation = _kalmanFilter.correct( measurement );
   //The corrected values get passed into the centering function which sets the (x,y) coordinates of the 
   //blur box appropriately
   CenterTrackWindowAboutPosition( static_cast< int >( estimation.at<float>( 0 ) ), static_cast< int >( estimation.at<float>( 1 ) ) );
}
