// A simple demo of JNI interface to implement SIFT detection for Android application using nonfree module in OpenCV4Android.
// Created by Guohui Wang 
// Email: robertwgh_at_gmail_com
// Data: 2/26/2014

#include <jni.h>
#include <android/log.h>

#include <opencv2/core/core.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/internal.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/calib3d/>
//#include <opencv2/calib3d/epnp.h>
//#include <opencv2/calib3d/p3p.h>
//#include <cvconfig.h>
#include <opencv2/photo/photo.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv/cxcore.h>
#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/features2d/features2d_tegra.hpp"
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace cv;
using namespace std;

#define  LOG_TAG    "opencv_auxiliaries"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

typedef unsigned char uchar;

cv::Mat ransacTest(cv::Mat& matches, cv::Mat& keypoints1, cv::Mat& keypoints2, cv::Mat& out_matches);
int little2big(int i);

extern "C" {
    JNIEXPORT jlong JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeRansacTest(JNIEnv * env, jobject, jlong matches_cptr, jlong keypoints1_cptr, jlong keypoints2_cptr, jlong result);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeInvertTranslation(JNIEnv * env, jobject, jlong rmat_cptr, jlong tvec_cptr, jlong T_cptr);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeWallisFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeGammaAdaptation(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jfloat gamma);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeWallisRGBFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeDrawMatchesCustom(JNIEnv * env, jobject, jlong img1Ptr, jlong kpImg1Ptr, jlong img2Ptr, jlong kpImg2Ptr, jlong matchesPtr, jlong imgTargetPtr, jdoubleArray matchColorPtr, jdoubleArray otherColorPtr, jint ptRadius, jint ptThickness, jint lineSize);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSERdetect(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSERdetectParameter(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jint delta, jint min_area, jint max_area, jdouble max_variation, jdouble min_diversity, jint max_evolution, jdouble area_threshold, jdouble min_margin, jint edge_blur_size);
    JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSCRSIFT(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jlong descriptorMatPtr);
    
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeWallisFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeGammaAdaptation(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jfloat gamma);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeWallisRGBFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeDrawMatchesCustom(JNIEnv * env, jobject, jlong img1Ptr, jlong kpImg1Ptr, jlong img2Ptr, jlong kpImg2Ptr, jlong matchesPtr, jlong imgTargetPtr, jdoubleArray matchColorPtr, jdoubleArray otherColorPtr, jint ptRadius, jint ptThickness, jint lineSize);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSERdetect(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSERdetectParameter(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jint delta, jint min_area, jint max_area, jdouble max_variation, jdouble min_diversity, jint max_evolution, jdouble area_threshold, jdouble min_margin, jint edge_blur_size);
    JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSCRSIFT(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jlong descriptorMatPtr);
};

cv::Vec3b getClampedValue(const cv::Mat& img, long x, long y, long width, long height)
{
	Vec3b result = Vec3b(0,0,0);
	if(width==-1)
		width = img.size().width;
	if(height==-1)
		height = img.size().height;
	if((x>0) && (x<width) && (y>0) && (y<height))
		result = img.at<Vec3b>(y,x);
	return result;
}
inline double sq (double a) {return a*a;}

// C++ / JNI
// vector_KeyPoint converters

void Mat_to_vector_KeyPoint(Mat& mat, vector<KeyPoint>& v_kp)
{
    v_kp.clear();
    assert(mat.type()==CV_32FC(7) && mat.cols==1);
    for(int i=0; i<mat.rows; i++)
    {
        Vec<float, 7> v = mat.at< Vec<float, 7> >(i, 0);
        KeyPoint kp(v[0], v[1], v[2], v[3], v[4], (int)v[5], (int)v[6]);
        v_kp.push_back(kp);
    }
    return;
}

void Mat_to_vector_DMatch(Mat& mat, vector<DMatch>& v_dm)
{
    v_dm.clear();
    assert(mat.type()==CV_32FC(4) && mat.cols==1);
    for(int i=0; i<mat.rows; i++)
    {
        Vec<float, 4> v = mat.at< Vec<float, 4> >(i, 0);
        DMatch dm((int)v[0], (int)v[1], (int)v[2], v[3]);
        v_dm.push_back(dm);
    }
    return;
}

void vector_KeyPoint_to_Mat(vector<KeyPoint>& v_kp, Mat& mat)
{
    int count = (int)v_kp.size();
    mat.create(count, 1, CV_32FC(7));
    for(int i=0; i<count; i++)
    {
        KeyPoint kp = v_kp[i];
        mat.at< Vec<float, 7> >(i, 0) = Vec<float, 7>(kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, (float)kp.octave, (float)kp.class_id);
    }
}

void vector_DMatch_to_Mat(vector<DMatch>& v_dm, Mat& mat)
{
    int count = (int)v_dm.size();
    mat.create(count, 1, CV_32FC(4));
    for(int i=0; i<count; i++)
    {
    	DMatch dm = v_dm[i];
        mat.at< Vec<float, 4> >(i, 0) = Vec<float, 4>((float)dm.queryIdx, (float)dm.trainIdx, (float)dm.imgIdx, dm.distance);
    }
}

JNIEXPORT jlong JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeRansacTest(JNIEnv * env, jobject, jlong matches_cptr, jlong keypoints1_cptr, jlong keypoints2_cptr, jlong result)
{
	LOGI("RANSAC fundamental matrix test \n");
	Mat& k1 = *(cv::Mat*)keypoints1_cptr;
	Mat& k2 = *(cv::Mat*)keypoints2_cptr;
	Mat& in_matches = *(cv::Mat*)matches_cptr;
	Mat& out_matches = *(cv::Mat*)result;
	ransacTest(in_matches, k1, k2, out_matches);
	LOGI("RANSAC fundamental matrix test!\n");
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeInvertTranslation(JNIEnv * env, jobject, jlong rmat_cptr, jlong tvec_cptr, jlong T_cptr)
{
	Mat R = *(Mat*)rmat_cptr;
	Mat tvec = *(Mat*)tvec_cptr;
	Mat* T_ptr = (Mat*)T_cptr;
	(*T_ptr) = -R*tvec;
	return;
}

//JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSERdetect(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr/*, jlong descriptorMatPtr*/) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	//Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	//Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector();
	detector->detect(image, keypoints);
	//delete detector;
	vector_KeyPoint_to_Mat(keypoints, keypointMat);
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSERdetect(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr/*, jlong descriptorMatPtr*/) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	//Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	//Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector();
	detector->detect(image, keypoints);
	//delete detector;
	vector_KeyPoint_to_Mat(keypoints, keypointMat);
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSERdetectParameter(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr/*, jlong descriptorMatPtr*/, jint delta, jint min_area, jint max_area, jdouble max_variation, jdouble min_diversity, jint max_evolution, jdouble area_threshold, jdouble min_margin, jint edge_blur_size) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	//Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	//Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
	detector->detect(image, keypoints);
	//delete detector;
	vector_KeyPoint_to_Mat(keypoints, keypointMat);
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSERdetectParameter(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr/*, jlong descriptorMatPtr*/, jint delta, jint min_area, jint max_area, jdouble max_variation, jdouble min_diversity, jint max_evolution, jdouble area_threshold, jdouble min_margin, jint edge_blur_size) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	//Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	//Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
	detector->detect(image, keypoints);
	//delete detector;
	vector_KeyPoint_to_Mat(keypoints, keypointMat);
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeMSCRSIFT(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jlong descriptorMatPtr) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector();
	detector->detect(image, keypoints);
	extractor = new SiftDescriptorExtractor();
	extractor->compute(image, keypoints, descriptorMat);
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeMSCRSIFT(JNIEnv* env, jobject, jlong imagePtr, jlong keypointMatPtr, jlong descriptorMatPtr) {
	Mat& image = *(cv::Mat*)imagePtr;
	Mat& keypointMat = *(cv::Mat*)keypointMatPtr;
	Mat& descriptorMat = *(cv::Mat*)descriptorMatPtr;
	vector<KeyPoint> keypoints;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	detector = new MserFeatureDetector();
	detector->detect(image, keypoints);
	extractor = new SiftDescriptorExtractor();
	extractor->compute(image, keypoints, descriptorMat);
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeDrawMatchesCustom(JNIEnv * env, jobject, jlong img1Ptr, jlong kpImg1Ptr, jlong img2Ptr, jlong kpImg2Ptr, jlong matchesPtr, jlong imgTargetPtr, jdoubleArray matchColorPtr, jdoubleArray otherColorPtr, jint ptRadius, jint ptThickness, jint lineSize)
{
	/*
	 * jsize size = env->GetArrayLength( arr );
std::vector<double> input( size );
env->GetDoubleArrayRegion( arr, 0, size, &input[0] );

//  ...

jdoubleArray output = env->NewDoubleArray( results.size() );
env->SetDoubleArrayRegion( output, 0, results.size(), &results[0] );
	 */
	Mat img1 = *(Mat*)img1Ptr;
	Mat img2 = *(Mat*)img2Ptr;
	Mat keypointsMat1 = *(Mat*)kpImg1Ptr;
	Mat keypointsMat2 = *(Mat*)kpImg2Ptr;
	Mat matches1to2Mat = *(Mat*)matchesPtr;
	Mat outImg = *(Mat*)imgTargetPtr;
	vector<KeyPoint> keypoints1;
	Mat_to_vector_KeyPoint(keypointsMat1, keypoints1);
	vector<KeyPoint> keypoints2;
	Mat_to_vector_KeyPoint(keypointsMat2, keypoints2);
	vector<DMatch> matches1to2;
	Mat_to_vector_DMatch(matches1to2Mat, matches1to2);
	jsize msize = env->GetArrayLength( matchColorPtr ), lsize = env->GetArrayLength( otherColorPtr );
	vector<double> mColor(msize), lColor(lsize);
	env->GetDoubleArrayRegion( matchColorPtr, 0, msize, &mColor[0] );
	env->GetDoubleArrayRegion( otherColorPtr, 0, lsize, &lColor[0] );
	Scalar matchColor = Scalar(mColor[0], mColor[1], mColor[2]);
	Scalar otherColor = Scalar(lColor[0], lColor[1], lColor[2]);

	Size sz1 = img1.size();
	Size sz2 = img2.size();
	//Mat(max(sz1.height, sz2.height), sz1.width+sz2.width, outImg.type()).copyTo(outImg);
	LOGI("DrawMatches - copied the re-created output image.");
	Mat left(outImg, cv::Rect(0, 0, sz1.width, sz1.height));
	img1.copyTo(left);
	LOGI("DrawMatches - copied the left ROI to output image.");
    Mat right(outImg, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
    img2.copyTo(right);
    LOGI("DrawMatches - copied the right ROI to output image.");

    //im3.adjustROI(0, 0, -sz1.width, sz2.width);
	for(uint i = 0; i < keypoints1.size(); i++)
	{
		circle(outImg, keypoints1.at(i).pt, ptRadius, otherColor,ptThickness, CV_AA, 0);
	}
	//im3.adjustROI(0, 0, sz1.width, 0);
	Point2f tPoint;
	for(uint i = 0; i < keypoints2.size(); i++)
	{
		tPoint = keypoints2.at(i).pt;
		tPoint.x+=float(sz1.width);
		circle(outImg, tPoint, ptRadius, otherColor,ptThickness, CV_AA, 0);
	}

	Point2f tOP;
	for(uint i = 0; i < matches1to2.size(); i++)
	{
		tOP = keypoints1.at(matches1to2.at(i).queryIdx).pt;
		tPoint = keypoints2.at(matches1to2.at(i).trainIdx).pt;
		tPoint.x+=float(sz1.width);
		circle(outImg, tOP, ptRadius, matchColor, -1, CV_AA, 0);
		circle(outImg, tPoint, ptRadius, matchColor, -1, CV_AA, 0);
		line(outImg, tOP, tPoint, matchColor, lineSize, CV_AA, 0);
	}
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Features_nativeDrawMatchesCustom(JNIEnv * env, jobject, jlong img1Ptr, jlong kpImg1Ptr, jlong img2Ptr, jlong kpImg2Ptr, jlong matchesPtr, jlong imgTargetPtr, jdoubleArray matchColorPtr, jdoubleArray otherColorPtr, jint ptRadius, jint ptThickness, jint lineSize)
{
	/*
	 * jsize size = env->GetArrayLength( arr );
std::vector<double> input( size );
env->GetDoubleArrayRegion( arr, 0, size, &input[0] );

//  ...

jdoubleArray output = env->NewDoubleArray( results.size() );
env->SetDoubleArrayRegion( output, 0, results.size(), &results[0] );
	 */
	Mat img1 = *(Mat*)img1Ptr;
	Mat img2 = *(Mat*)img2Ptr;
	Mat keypointsMat1 = *(Mat*)kpImg1Ptr;
	Mat keypointsMat2 = *(Mat*)kpImg2Ptr;
	Mat matches1to2Mat = *(Mat*)matchesPtr;
	Mat outImg = *(Mat*)imgTargetPtr;
	vector<KeyPoint> keypoints1;
	Mat_to_vector_KeyPoint(keypointsMat1, keypoints1);
	vector<KeyPoint> keypoints2;
	Mat_to_vector_KeyPoint(keypointsMat2, keypoints2);
	vector<DMatch> matches1to2;
	Mat_to_vector_DMatch(matches1to2Mat, matches1to2);
	jsize msize = env->GetArrayLength( matchColorPtr ), lsize = env->GetArrayLength( otherColorPtr );
	vector<double> mColor(msize), lColor(lsize);
	env->GetDoubleArrayRegion( matchColorPtr, 0, msize, &mColor[0] );
	env->GetDoubleArrayRegion( otherColorPtr, 0, lsize, &lColor[0] );
	Scalar matchColor = Scalar(mColor[0], mColor[1], mColor[2]);
	Scalar otherColor = Scalar(lColor[0], lColor[1], lColor[2]);

	Size sz1 = img1.size();
	Size sz2 = img2.size();
	//Mat(max(sz1.height, sz2.height), sz1.width+sz2.width, outImg.type()).copyTo(outImg);
	LOGI("DrawMatches - copied the re-created output image.");
	Mat left(outImg, cv::Rect(0, 0, sz1.width, sz1.height));
	img1.copyTo(left);
	LOGI("DrawMatches - copied the left ROI to output image.");
    Mat right(outImg, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
    img2.copyTo(right);
    LOGI("DrawMatches - copied the right ROI to output image.");

    //im3.adjustROI(0, 0, -sz1.width, sz2.width);
	for(uint i = 0; i < keypoints1.size(); i++)
	{
		circle(outImg, keypoints1.at(i).pt, ptRadius, otherColor,ptThickness, CV_AA, 0);
	}
	//im3.adjustROI(0, 0, sz1.width, 0);
	Point2f tPoint;
	for(uint i = 0; i < keypoints2.size(); i++)
	{
		tPoint = keypoints2.at(i).pt;
		tPoint.x+=float(sz1.width);
		circle(outImg, tPoint, ptRadius, otherColor,ptThickness, CV_AA, 0);
	}

	Point2f tOP;
	for(uint i = 0; i < matches1to2.size(); i++)
	{
		tOP = keypoints1.at(matches1to2.at(i).queryIdx).pt;
		tPoint = keypoints2.at(matches1to2.at(i).trainIdx).pt;
		tPoint.x+=float(sz1.width);
		circle(outImg, tOP, ptRadius, matchColor, -1, CV_AA, 0);
		circle(outImg, tPoint, ptRadius, matchColor, -1, CV_AA, 0);
		line(outImg, tOP, tPoint, matchColor, lineSize, CV_AA, 0);
	}
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeWallisFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Wallis adaptation.");
	}

	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = (cv::Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = (cv::Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	float targetStDev = 50.0f; 
	float targetMean = 256.0f; 
	float alfa = 0.0f;
	float limit = 10.0f;

	Mat ycrcbIn = Mat::zeros(imgIn.size(), CV_8UC3);
	Mat ycrcbOut = Mat::zeros(imgIn.size(), CV_8UC3);
	Mat(imgIn.size().height, imgIn.size().width, CV_8UC3).copyTo(outImg);
	cvtColor(imgIn, ycrcbIn, CV_BGR2YCrCb);

	/*
	 * Original Wallis - just on Y-channel
	 */
	Vec3b pix, sout;
	long half = ((int)kernelSize - 1) / 2;
	long w = imgIn.size().width, h = imgIn.size().height;
	float mY, stdY;
	long c,r;
	long xt, yt;
	//int cg = 1; // Wallis standard value
	//float b = 1.5; // Wallis standard value
	//float r1, r0; // Wallis shift and scale parameters
	int size = (int)kernelSize*(int)kernelSize;
	long c_start, c_end, r_start, r_end;
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
	    mY=0;
	    c_start = x - half; r_start = y - half;
	    c_end = x + half; r_end = y + half;
	    for (c = c_start; c < c_end; c++){
	    	for (r = r_start; r < r_end; r++){
	    	  pix = getClampedValue(ycrcbIn, c,r,w,h);
	    	  mY += pix.val[0];
	    	}
	    }
	    mY = mY / size;
	    stdY=0;
	    for (c = c_start; c < c_end; c++){
	      for (r = r_start; r < r_end; r++){
	    	  pix = getClampedValue(ycrcbIn, c,r,w,h);
	    	  stdY += sq(pix.val[0]-mY);
	      }
	    }
	    stdY = sqrt(stdY / size);

	    //Calc new values
	    xt = x; yt = y;
	    pix = ycrcbIn.at<Vec3b>(yt,xt);

	    //r1 = cg * to_dev / (cg * stdB + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mB;
	    //sout.val[0] = pixB.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[0] = (int)(alfa * pix.val[0] + (1-alfa) * mY + (pix.val[0]-mY) * targetStDev / (targetStDev/limit+stdY));
	    else
	    	sout.val[0] = (int)(alfa * targetMean + (1-alfa) * mY + (pix.val[0]-mY) * targetStDev / (targetStDev/limit+stdY));

	    sout.val[1] = pix.val[1];
	    sout.val[2] = pix.val[2];
	    // Write new output value
	    ycrcbOut.at<Vec3b>(y, x).val[0] = sout.val[0];
	    ycrcbOut.at<Vec3b>(y, x).val[1] = sout.val[1];
	    ycrcbOut.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}

	Mat bgrOut = Mat::zeros(outImg.size(), CV_8UC3);
	cvtColor(ycrcbOut, bgrOut, CV_YCrCb2BGR);
	bgrOut.copyTo(outImg);
	ycrcbIn.release();
	ycrcbOut.release();
	bgrOut.release();
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeWallisFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Wallis adaptation.");
	}

	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = (cv::Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = (cv::Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	float targetStDev = 50.0f; 
	float targetMean = 256.0f; 
	float alfa = 0.0f;
	float limit = 10.0f;

	Mat ycrcbIn = Mat::zeros(imgIn.size(), CV_8UC3);
	Mat ycrcbOut = Mat::zeros(imgIn.size(), CV_8UC3);
	Mat(imgIn.size().height, imgIn.size().width, CV_8UC3).copyTo(outImg);
	cvtColor(imgIn, ycrcbIn, CV_BGR2YCrCb);

	/*
	 * Original Wallis - just on Y-channel
	 */
	Vec3b pix, sout;
	long half = ((int)kernelSize - 1) / 2;
	long w = imgIn.size().width, h = imgIn.size().height;
	float mY, stdY;
	long c,r;
	long xt, yt;
	//int cg = 1; // Wallis standard value
	//float b = 1.5; // Wallis standard value
	//float r1, r0; // Wallis shift and scale parameters
	int size = (int)kernelSize*(int)kernelSize;
	long c_start, c_end, r_start, r_end;
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
	    mY=0;
	    c_start = x - half; r_start = y - half;
	    c_end = x + half; r_end = y + half;
	    for (c = c_start; c < c_end; c++){
	    	for (r = r_start; r < r_end; r++){
	    	  pix = getClampedValue(ycrcbIn, c,r,w,h);
	    	  mY += pix.val[0];
	    	}
	    }
	    mY = mY / size;
	    stdY=0;
	    for (c = c_start; c < c_end; c++){
	      for (r = r_start; r < r_end; r++){
	    	  pix = getClampedValue(ycrcbIn, c,r,w,h);
	    	  stdY += sq(pix.val[0]-mY);
	      }
	    }
	    stdY = sqrt(stdY / size);

	    //Calc new values
	    xt = x; yt = y;
	    pix = ycrcbIn.at<Vec3b>(yt,xt);

	    //r1 = cg * to_dev / (cg * stdB + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mB;
	    //sout.val[0] = pixB.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[0] = (int)(alfa * pix.val[0] + (1-alfa) * mY + (pix.val[0]-mY) * targetStDev / (targetStDev/limit+stdY));
	    else
	    	sout.val[0] = (int)(alfa * targetMean + (1-alfa) * mY + (pix.val[0]-mY) * targetStDev / (targetStDev/limit+stdY));

	    sout.val[1] = pix.val[1];
	    sout.val[2] = pix.val[2];
	    // Write new output value
	    ycrcbOut.at<Vec3b>(y, x).val[0] = sout.val[0];
	    ycrcbOut.at<Vec3b>(y, x).val[1] = sout.val[1];
	    ycrcbOut.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}

	Mat bgrOut = Mat::zeros(outImg.size(), CV_8UC3);
	cvtColor(ycrcbOut, bgrOut, CV_YCrCb2BGR);
	bgrOut.copyTo(outImg);
	ycrcbIn.release();
	ycrcbOut.release();
	bgrOut.release();
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeWallisRGBFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Wallis adaptation.");
	}

	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = (cv::Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = (cv::Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	float targetStDev = 50.0f;
	float targetMean = 256.0f;
	float alfa = 0.0f;
	float limit = 10.0f;

	Mat(imgIn.size().height, imgIn.size().width, CV_8UC3).copyTo(outImg);

	/*
	 * Original Wallis - just on Y-channel
	 */
	Vec3b pix, sout;
	long half = (kernelSize - 1) / 2;
	//long x_end = imgIn.size().width - kernelSize;
	//long y_end = imgIn.size().height - kernelSize;
	long w = imgIn.size().width, h = imgIn.size().height;
	float mR, mG, mB, stdR, stdG, stdB;
	long c,r;
	long xt, yt;
	//int cg = 1; // Wallis standard value
	//float b = 1.5; // Wallis standard value
	//float r1, r0; // Wallis shift and scale parameters
	int size = kernelSize*kernelSize;
	long c_start, c_end, r_start, r_end;
	//for (long x = 0; x < x_end; x++){
	//  for (long y = 0; y < y_end; y++){
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
	    mR=mG=mB=0;
	    c_start = x - half; r_start = y - half;
	    c_end = x + half; r_end = y + half;
	    //c_end = x + (kernelSize - 1); r_end = y + kernelSize - 1;
	    //for (c = x; c < c_end; c++){
	    //  for (r = y; r < r_end; r++){
	    for (c = c_start; c < c_end; c++){
	    	for (r = r_start; r < r_end; r++){
	    	  //imgIn.at<cv::Vec3b>(r,c).val[0];
	    	  //pix = imgIn.at<cv::Vec3b>(r,c);
	    	  pix = getClampedValue(imgIn, c,r,w,h);
	    	  mR += pix.val[2];
	    	  mG += pix.val[1];
	    	  mB += pix.val[0];
	    	}
	    }
	    mR = mR / size;
	    mG = mG / size;
	    mB = mB / size;
	    stdR=stdB=stdG=0;
	    //for (c = x; c < c_end; c++){
	    //  for (r = y; r < r_end; r++){
	    for (c = c_start; c < c_end; c++){
	      for (r = r_start; r < r_end; r++){
	    	  //pix = imgIn.at<cv::Vec3b>(r,c);
	    	  pix = getClampedValue(imgIn, c,r,w,h);
	    	  stdR += sq(pix.val[2]-mR);
	    	  stdG += sq(pix.val[1]-mG);
	    	  stdB += sq(pix.val[0]-mB);
	      }
	    }
	    stdB = sqrt(stdB / size);
	    stdG = sqrt(stdG / size);
	    stdR = sqrt(stdR / size);

	    //Calc new values
	    //xt = x+half+1; yt = y+half+1;
	    //xt = x+half; yt = y+half;
	    xt = x; yt = y;
	    pix = imgIn.at<cv::Vec3b>(yt,xt);

	    //r1 = cg * to_dev / (cg * stdB + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mB;
	    //sout.val[0] = pixB.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[0] = alfa * pix.val[0] + (1-alfa) * mB + (pix.val[0]-mB) * targetStDev / (targetStDev/limit+stdB);
	    else
	    	sout.val[0] = alfa * targetMean + (1-alfa) * mB + (pix.val[0]-mB) * targetStDev / (targetStDev/limit+stdB);


	    //r1 = cg * to_dev / (cg * stdG + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mG;
	    //sout.val[1] = pixG.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[1] = alfa * pix.val[1] + (1-alfa) * mG + (pix.val[1]-mG) * targetStDev / (targetStDev/limit+stdG);
	    else
	    	sout.val[1] = alfa * targetMean + (1-alfa) * mG + (pix.val[1]-mG) * targetStDev / (targetStDev/limit+stdG);

	    //r1 = cg * to_dev / (cg * stdR + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mR;
	    //sout.val[2] = pixR.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[2] = alfa * pix.val[2] + (1-alfa) * mR + (pix.val[2]-mR) * targetStDev / (targetStDev/limit+stdR);
	    else
	    	sout.val[2] = alfa * targetMean + (1-alfa) * mR + (pix.val[2]-mR) * targetStDev / (targetStDev/limit+stdR);

	    // Write new output value
	    outImg.at<Vec3b>(y, x).val[0] = sout.val[0];
	    outImg.at<Vec3b>(y, x).val[1] = sout.val[1];
	    outImg.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeWallisRGBFilter(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jint kernelSize) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Wallis adaptation.");
	}

	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = (cv::Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = (cv::Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	float targetStDev = 50.0f;
	float targetMean = 256.0f;
	float alfa = 0.0f;
	float limit = 10.0f;

	Mat(imgIn.size().height, imgIn.size().width, CV_8UC3).copyTo(outImg);

	/*
	 * Original Wallis - just on Y-channel
	 */
	Vec3b pix, sout;
	long half = (kernelSize - 1) / 2;
	//long x_end = imgIn.size().width - kernelSize;
	//long y_end = imgIn.size().height - kernelSize;
	long w = imgIn.size().width, h = imgIn.size().height;
	float mR, mG, mB, stdR, stdG, stdB;
	long c,r;
	long xt, yt;
	//int cg = 1; // Wallis standard value
	//float b = 1.5; // Wallis standard value
	//float r1, r0; // Wallis shift and scale parameters
	int size = kernelSize*kernelSize;
	long c_start, c_end, r_start, r_end;
	//for (long x = 0; x < x_end; x++){
	//  for (long y = 0; y < y_end; y++){
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
	    mR=mG=mB=0;
	    c_start = x - half; r_start = y - half;
	    c_end = x + half; r_end = y + half;
	    //c_end = x + (kernelSize - 1); r_end = y + kernelSize - 1;
	    //for (c = x; c < c_end; c++){
	    //  for (r = y; r < r_end; r++){
	    for (c = c_start; c < c_end; c++){
	    	for (r = r_start; r < r_end; r++){
	    	  //imgIn.at<cv::Vec3b>(r,c).val[0];
	    	  //pix = imgIn.at<cv::Vec3b>(r,c);
	    	  pix = getClampedValue(imgIn, c,r,w,h);
	    	  mR += pix.val[2];
	    	  mG += pix.val[1];
	    	  mB += pix.val[0];
	    	}
	    }
	    mR = mR / size;
	    mG = mG / size;
	    mB = mB / size;
	    stdR=stdB=stdG=0;
	    //for (c = x; c < c_end; c++){
	    //  for (r = y; r < r_end; r++){
	    for (c = c_start; c < c_end; c++){
	      for (r = r_start; r < r_end; r++){
	    	  //pix = imgIn.at<cv::Vec3b>(r,c);
	    	  pix = getClampedValue(imgIn, c,r,w,h);
	    	  stdR += sq(pix.val[2]-mR);
	    	  stdG += sq(pix.val[1]-mG);
	    	  stdB += sq(pix.val[0]-mB);
	      }
	    }
	    stdB = sqrt(stdB / size);
	    stdG = sqrt(stdG / size);
	    stdR = sqrt(stdR / size);

	    //Calc new values
	    //xt = x+half+1; yt = y+half+1;
	    //xt = x+half; yt = y+half;
	    xt = x; yt = y;
	    pix = imgIn.at<cv::Vec3b>(yt,xt);

	    //r1 = cg * to_dev / (cg * stdB + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mB;
	    //sout.val[0] = pixB.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[0] = alfa * pix.val[0] + (1-alfa) * mB + (pix.val[0]-mB) * targetStDev / (targetStDev/limit+stdB);
	    else
	    	sout.val[0] = alfa * targetMean + (1-alfa) * mB + (pix.val[0]-mB) * targetStDev / (targetStDev/limit+stdB);


	    //r1 = cg * to_dev / (cg * stdG + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mG;
	    //sout.val[1] = pixG.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[1] = alfa * pix.val[1] + (1-alfa) * mG + (pix.val[1]-mG) * targetStDev / (targetStDev/limit+stdG);
	    else
	    	sout.val[1] = alfa * targetMean + (1-alfa) * mG + (pix.val[1]-mG) * targetStDev / (targetStDev/limit+stdG);

	    //r1 = cg * to_dev / (cg * stdR + to_dev / cg);
	    //r0 = b * to_av + (1 - b - r1) * mR;
	    //sout.val[2] = pixR.val[0] * r1 + r0 ;
	    // HIPS implementation
	    if(int(targetMean)==256)
	    	sout.val[2] = alfa * pix.val[2] + (1-alfa) * mR + (pix.val[2]-mR) * targetStDev / (targetStDev/limit+stdR);
	    else
	    	sout.val[2] = alfa * targetMean + (1-alfa) * mR + (pix.val[2]-mR) * targetStDev / (targetStDev/limit+stdR);

	    // Write new output value
	    outImg.at<Vec3b>(y, x).val[0] = sout.val[0];
	    outImg.at<Vec3b>(y, x).val[1] = sout.val[1];
	    outImg.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}
}

JNIEXPORT void JNICALL Java_christian_fragmentexample_AutoRegistrationFragment_nativeGammaAdaptation(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jfloat gamma) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Gamma adaptation.");
	}
	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	Mat ycrcbIn = cv::Mat::zeros(imgIn.size(), CV_8UC3);
	Mat ycrcbOut = cv::Mat::zeros(imgIn.size(), CV_8UC3);
	cvtColor(imgIn, ycrcbIn, CV_BGR2YCrCb);

	cv::Vec3b pix, sout;
	long w = imgIn.size().width, h = imgIn.size().height;
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
		pix = getClampedValue(ycrcbIn, x,y,w,h);
		float value = (float(pix.val[0])/255.0f);
		value = max(0.0f, min(1.0f, pow(value, gamma)));
	    sout.val[0] = int(value*255.0f);
	    sout.val[1] = pix.val[1];
	    sout.val[2] = pix.val[2];
	    // Write new output value
	    ycrcbOut.at<Vec3b>(y, x).val[0] = sout.val[0];
	    ycrcbOut.at<Vec3b>(y, x).val[1] = sout.val[1];
	    ycrcbOut.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}

	Mat bgrOut = Mat::zeros(outImg.size(), CV_8UC3);
	cvtColor(ycrcbOut, bgrOut, CV_YCrCb2BGR);
	bgrOut.copyTo(outImg);
	//for (long x = 0; x < w; x++){
	//  for (long y = 0; y < h; y++){
	//	  outImg.at<cv::Vec3b>(y,x) = bgrOut.at<cv::Vec3b>(y,x);
	//  }
	//}
	ycrcbIn.release();
	ycrcbOut.release();
	bgrOut.release();
	LOGI("finished Gamma adaptation.");
}

JNIEXPORT void JNICALL Java_org_opencv_auxiliary_Filters_nativeGammaAdaptation(JNIEnv* env, jobject, jlong imgInMatPtr, jlong imgOutMatPtr, jfloat gamma) {
	if((imgInMatPtr==0) || (imgOutMatPtr==0)){
		LOGE("in- or out image are NULL.");
		return;
	} else {
		LOGI("starting Gamma adaptation.");
	}
	Mat& imgIn = *(Mat*)imgInMatPtr;
	//cv::Mat* imgIn = reinterpret_cast<cv::Mat*>(imgInMatPtr);
	Mat& outImg = *(Mat*)imgOutMatPtr;
	//cv::Mat* outImg = reinterpret_cast<cv::Mat*>(imgOutMatPtr);


	Mat ycrcbIn = cv::Mat::zeros(imgIn.size(), CV_8UC3);
	Mat ycrcbOut = cv::Mat::zeros(imgIn.size(), CV_8UC3);
	cvtColor(imgIn, ycrcbIn, CV_BGR2YCrCb);

	cv::Vec3b pix, sout;
	long w = imgIn.size().width, h = imgIn.size().height;
	for (long x = 0; x < w; x++){
	  for (long y = 0; y < h; y++){
		// compute statistics
		pix = getClampedValue(ycrcbIn, x,y,w,h);
		float value = (float(pix.val[0])/255.0f);
		value = max(0.0f, min(1.0f, pow(value, gamma)));
	    sout.val[0] = int(value*255.0f);
	    sout.val[1] = pix.val[1];
	    sout.val[2] = pix.val[2];
	    // Write new output value
	    ycrcbOut.at<Vec3b>(y, x).val[0] = sout.val[0];
	    ycrcbOut.at<Vec3b>(y, x).val[1] = sout.val[1];
	    ycrcbOut.at<Vec3b>(y, x).val[2] = sout.val[2];
	  }
	}

	Mat bgrOut = Mat::zeros(outImg.size(), CV_8UC3);
	cvtColor(ycrcbOut, bgrOut, CV_YCrCb2BGR);
	bgrOut.copyTo(outImg);
	//for (long x = 0; x < w; x++){
	//  for (long y = 0; y < h; y++){
	//	  outImg.at<cv::Vec3b>(y,x) = bgrOut.at<cv::Vec3b>(y,x);
	//  }
	//}
	ycrcbIn.release();
	ycrcbOut.release();
	bgrOut.release();
	LOGI("finished Gamma adaptation.");
}

/*
cv::Mat findFundamentalMat(cv::Mat& matches, cv::Mat& keypoints1, cv::Mat& keypoints2, cv::Mat& out_matches)
{
	std::vector<uchar> inliers(points1.size(),0);
	cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), // matching points
				inliers,       // match status (inlier or outlier)
				CV_FM_RANSAC, // RANSAC method
				7.0f,      // distance to epipolar line
				0.97f); // confidence probability
	
}
*/

cv::Mat ransacTest(cv::Mat& matches, cv::Mat& keypoints1, cv::Mat& keypoints2, cv::Mat& out_matches)
{
	// Convert keypoints into Point2f
	int refineF = 0;
	LOGI("input Matrices to vectors ...");
	std::vector<cv::DMatch> outMatches;
	LOGI("matches ... size: %i x %i", matches.rows, matches.cols);
	std::vector<cv::DMatch> matches_list; 
	//matches.copyTo(matches_list);
	//matches_list.assign((cv::DMatch*)matches.datastart, (cv::DMatch*)matches.dataend);
	matches_list.resize(matches.rows);
	memcpy(matches_list.data(), matches.data, matches.rows*sizeof(cv::DMatch));
	std::vector<cv::KeyPoint> klist_1, klist_2; 
	LOGI("keypoints 1 ... size: %i x %i", keypoints1.rows, keypoints1.cols);
	//keypoints1.copyTo(klist_1);
	//klist_1.assign((cv::KeyPoint*)keypoints1.datastart, (cv::KeyPoint*)keypoints1.dataend);
	klist_1.resize(keypoints1.rows);
	memcpy(klist_1.data(), keypoints1.data, keypoints1.rows*sizeof(cv::KeyPoint));
	LOGI("keypoints 2 ... size: %i x %i", keypoints2.rows, keypoints2.cols);
	//keypoints2.copyTo(klist_2);
	//klist_2.assign((cv::KeyPoint*)keypoints2.datastart, (cv::KeyPoint*)keypoints2.dataend);
	klist_2.resize(keypoints2.rows);
	memcpy(klist_2.data(), keypoints2.data, keypoints2.rows*sizeof(cv::KeyPoint));
	std::vector<cv::Point2f> points1, points2;
	LOGI("DONE.");
	cv::Mat fundemental;
	for (std::vector<cv::DMatch>::iterator it= matches_list.begin(); it!= matches_list.end(); it++) {
		LOGI("Element %i => %i", little2big(it->queryIdx), little2big(it->trainIdx));
		// Get the position of left keypoints
		float x= klist_1[little2big(it->queryIdx)].pt.x;
		float y= klist_1[little2big(it->queryIdx)].pt.y;
		points1.push_back(cv::Point2f(x,y));
		// Get the position of right keypoints
		x= klist_2[little2big(it->trainIdx)].pt.x;
		y= klist_2[little2big(it->trainIdx)].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}
	LOGI("Points copied ..."); 
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(),0);
	if (points1.size()>0&&points2.size()>0){
		cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), // matching points
				inliers,       // match status (inlier or outlier)
				CV_FM_RANSAC, // RANSAC method
				7.0f,      // distance to epipolar line
				0.97f); // confidence probability
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM= matches_list.begin();
		// for all matches
		for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		LOGI("Inlier retrieved.");
		if (refineF) {
			// The F matrix will be recomputed with
			// all accepted matches
			// Convert keypoints into Point2f
			// for final F computation
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin(); it!= outMatches.end(); ++it) {
				// Get the position of left keypoints
				float x= klist_1[it->queryIdx].pt.x;
				float y= klist_1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));
				// Get the position of right keypoints
				x= klist_2[it->trainIdx].pt.x;
				y= klist_2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
			}

			// Compute 8-point F from all accepted matches
			if (points1.size()>0&&points2.size()>0){
				fundemental= cv::findFundamentalMat(
						cv::Mat(points1),cv::Mat(points2), // matches
						CV_FM_8POINT); // 8-point method
			}
		}
	}
	LOGI("output Matrices to vectors ...");
	//cv::Mat(outMatches,true).copyTo(out_matches);
	out_matches.resize(outMatches.size());
	memcpy(out_matches.data, outMatches.data(), outMatches.size()*sizeof(cv::DMatch));
	LOGI("DONE.");
	return fundemental;
}

int little2big(int i) {
    return (i&0xff)<<24 | (i&0xff00)<<8 | (i&0xff0000)>>8 | (i>>24)&0xff;
}
