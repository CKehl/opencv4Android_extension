LOCAL_PATH  := $(call my-dir)
OPENCV_PATH := /media/christian/DATA/OCV_Android_2410/OpenCV-2.4.10-android-sdk/sdk/native/jni

include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES := on
OPENCV_CAMERA_MODULES  := off
include $(OPENCV_PATH)/OpenCV.mk

LOCAL_C_INCLUDES :=				\
	$(LOCAL_PATH)				\
	$(OPENCV_PATH)/include

LOCAL_SRC_FILES :=				\
	auxiliary_jni.cpp
	
LOCAL_MODULE := opencv_auxiliaries
LOCAL_CFLAGS := -Werror -O3 -ffast-math
LOCAL_LDLIBS := -llog -ldl

include $(BUILD_SHARED_LIBRARY)
