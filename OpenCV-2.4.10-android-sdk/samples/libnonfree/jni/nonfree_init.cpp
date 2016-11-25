/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(SURF, "Feature2D.SURF",
                  obj.info()->addParam(obj, "hessianThreshold", obj.hessianThreshold);
                  obj.info()->addParam(obj, "nOctaves", obj.nOctaves);
                  obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
                  obj.info()->addParam(obj, "extended", obj.extended);
                  obj.info()->addParam(obj, "upright", obj.upright))

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(SIFT, "Feature2D.SIFT",
                  obj.info()->addParam(obj, "nfeatures", obj.nfeatures);
                  obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
                  obj.info()->addParam(obj, "contrastThreshold", obj.contrastThreshold);
                  obj.info()->addParam(obj, "edgeThreshold", obj.edgeThreshold);
                  obj.info()->addParam(obj, "sigma", obj.sigma))

CV_INIT_ALGORITHM(MSER, "Feature2D.MSER",
                  obj.info()->addParam(obj, "delta", obj.delta);
                  obj.info()->addParam(obj, "minArea", obj.minArea);
                  obj.info()->addParam(obj, "maxArea", obj.maxArea);
                  obj.info()->addParam(obj, "maxVariation", obj.maxVariation);
                  obj.info()->addParam(obj, "minDiversity", obj.minDiversity);
                  obj.info()->addParam(obj, "maxEvolution", obj.maxEvolution);
                  obj.info()->addParam(obj, "areaThreshold", obj.areaThreshold);
                  obj.info()->addParam(obj, "minMargin", obj.minMargin);
                  obj.info()->addParam(obj, "edgeBlurSize", obj.edgeBlurSize))

#ifdef HAVE_OPENCV_OCL
//
//namespace ocl {
//CV_INIT_ALGORITHM(SURF_OCL, "Feature2D.SURF_OCL",
//                  obj.info()->addParam(obj, "hessianThreshold", obj.hessianThreshold);
//                  obj.info()->addParam(obj, "nOctaves", obj.nOctaves);
//                  obj.info()->addParam(obj, "nOctaveLayers", obj.nOctaveLayers);
//                  obj.info()->addParam(obj, "extended", obj.extended);
//                  obj.info()->addParam(obj, "upright", obj.upright))
//}
//
#endif


///////////////////////////////////////////////////////////////////////////////////////////////////////////

bool initModule_nonfree(void)
{
    Ptr<Algorithm> sift = createSIFT(), surf = createSURF(), mser = createMSER();
    return sift->info() != 0 && surf->info() != 0;
}

}
