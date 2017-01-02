package org.opencv.auxiliary;

import org.opencv.core.Mat;

public class Filters extends Object {
    private static native void nativeWallisFilter(long imgInMatPtr, long imgOutMatPtr, int kernelSize);
    private static native void nativeGammaAdaptation(long imgInMatPtr, long imgOutMatPtr, float gamma);
    private static native void nativeWallisRGBFilter(long imgInMatPtr, long imgOutMatPtr, int kernelSize);
    
    public Filters() {
    	
    }
    
    public static void Wallis(Mat src, Mat dst, int kernelSize) {
    	nativeWallisFilter(src.getNativeObjAddr(), dst.getNativeObjAddr(), kernelSize);
    }
    
    public static void GammaAdaptation(Mat src, Mat dst, float gamma) {
    	nativeGammaAdaptation(src.getNativeObjAddr(), dst.getNativeObjAddr(), gamma);
    }
    
    public static void WallisRGB(Mat src, Mat dst, int kernelSize) {
    	nativeWallisRGBFilter(src.getNativeObjAddr(), dst.getNativeObjAddr(), kernelSize);
    }
};