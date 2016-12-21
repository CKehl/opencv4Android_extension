LOCAL_PATH  := $(call my-dir)
OPENCV_PATH := /media/christian/DATA/OCV_Android_2410/OpenCV-2.4.10-android-sdk/sdk/native/jni
NONFREE_LIBS_DIR :=/media/christian/DATA/git/OpenCV4Android/OpenCV-2.4.10-android-sdk/samples/libnonfree/libs/$(TARGET_ARCH_ABI)

define add_nonfree_module
    include $(CLEAR_VARS)
    LOCAL_MODULE:=nonfree
    LOCAL_SRC_FILES:=$(NONFREE_LIBS_DIR)/libnonfree.so
    include $(PREBUILT_SHARED_LIBRARY)
endef

$(eval $(call add_nonfree_module))
#LOCAL_SHARED_LIBRARIES += nonfree

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
LOCAL_LDLIBS := -llog -ldl #-lnonfree 
#LOCAL_LDFLAGS := -L$(NONFREE_LIBS_DIR)
LOCAL_SHARED_LIBRARIES += nonfree
# -L$(NONFREE_LIBS_DIR) -lnonfree

include $(BUILD_SHARED_LIBRARY)
