package org.opencv.auxiliary;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;

public class Features extends Object {
    private static native void nativeDrawMatchesCustom(long img1Ptr, long kpImg1Ptr, long img2Ptr, long kpImg2Ptr, long matchesPtr, long imgTargetPtr, double[] matchColorPtr, double[] otherColorPtr, int ptRadius, int ptThickness, int lineSize);
    private static native void nativeMSERdetect(long imagePtr, long keypointMatPtr);
    private static native void nativeMSERdetectParameter(long imagePtr, long keypointMatPtr, int delta, int min_area, int max_area, double max_variation, double min_diversity, int max_evolution, double area_threshold, double min_margin, int edge_blur_size);
    private static native void nativeMSCRSIFT(long imagePtr, long keypointMatPtr, long descriptorMatPtr);
	
    public Features() {
    	
    }
    
    public static void drawMatchesCustom(Mat img1, MatOfKeyPoint keypointsImg1, Mat img2, MatOfKeyPoint keypointsImg2, MatOfDMatch matches, Mat matchImg, Scalar pointColor, Scalar lineColor, int pointRadius, int pointThickness, int lineThickness) {
    	nativeDrawMatchesCustom(img1.getNativeObjAddr(), keypointsImg1.getNativeObjAddr(), img2.getNativeObjAddr(), keypointsImg2.getNativeObjAddr(), matches.getNativeObjAddr(), matchImg.getNativeObjAddr(), pointColor.val, lineColor.val, pointRadius, pointThickness, lineThickness);
    }
    
    public static void MSERdetect(Mat img, MatOfKeyPoint keypoints_dst) {
    	nativeMSERdetect(img.getNativeObjAddr(), keypoints_dst.getNativeObjAddr());
    }
    
    public static void MSERdetect(Mat img, MatOfKeyPoint keypoints_dst, int delta, int min_area, int max_area, double max_variation, double min_diversity, int max_evolution, double area_threshold, double min_margin, int edge_blur_size) {
    	nativeMSERdetectParameter(img.getNativeObjAddr(), keypoints_dst.getNativeObjAddr(), delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
    }
    
    public static void MSCRSIFT(Mat img, MatOfKeyPoint keypoints_dst, Mat descriptor_dst) {
    	nativeMSCRSIFT(img.getNativeObjAddr(), keypoints_dst.getNativeObjAddr(), descriptor_dst.getNativeObjAddr());
    }
};