# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. Example variables are:
#   CPACK_GENERATOR                     - Generator used to create package
#   CPACK_INSTALL_CMAKE_PROJECTS        - For each project (path, name, component)
#   CPACK_CMAKE_GENERATOR               - CMake Generator used for the projects
#   CPACK_INSTALL_COMMANDS              - Extra commands to install components
#   CPACK_INSTALLED_DIRECTORIES           - Extra directories to install
#   CPACK_PACKAGE_DESCRIPTION_FILE      - Description file for the package
#   CPACK_PACKAGE_DESCRIPTION_SUMMARY   - Summary of the package
#   CPACK_PACKAGE_EXECUTABLES           - List of pairs of executables and labels
#   CPACK_PACKAGE_FILE_NAME             - Name of the package generated
#   CPACK_PACKAGE_ICON                  - Icon used for the package
#   CPACK_PACKAGE_INSTALL_DIRECTORY     - Name of directory for the installer
#   CPACK_PACKAGE_NAME                  - Package project name
#   CPACK_PACKAGE_VENDOR                - Package project vendor
#   CPACK_PACKAGE_VERSION               - Package project version
#   CPACK_PACKAGE_VERSION_MAJOR         - Package project version (major)
#   CPACK_PACKAGE_VERSION_MINOR         - Package project version (minor)
#   CPACK_PACKAGE_VERSION_PATCH         - Package project version (patch)

# There are certain generator specific ones

# NSIS Generator:
#   CPACK_PACKAGE_INSTALL_REGISTRY_KEY  - Name of the registry key for the installer
#   CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS - Extra commands used during uninstall
#   CPACK_NSIS_EXTRA_INSTALL_COMMANDS   - Extra commands used during install


SET(CPACK_BINARY_BUNDLE "")
SET(CPACK_BINARY_CYGWIN "")
SET(CPACK_BINARY_DEB "OFF")
SET(CPACK_BINARY_DRAGNDROP "")
SET(CPACK_BINARY_NSIS "OFF")
SET(CPACK_BINARY_OSXX11 "")
SET(CPACK_BINARY_PACKAGEMAKER "")
SET(CPACK_BINARY_RPM "OFF")
SET(CPACK_BINARY_STGZ "ON")
SET(CPACK_BINARY_TBZ2 "OFF")
SET(CPACK_BINARY_TGZ "ON")
SET(CPACK_BINARY_TZ "ON")
SET(CPACK_BINARY_ZIP "")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")
SET(CPACK_COMPONENTS_ALL "dev;java;libs")
SET(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
SET(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
SET(CPACK_COMPONENT_dev_DEPENDS "libs")
SET(CPACK_COMPONENT_dev_DESCRIPTION "Development files for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_dev_DISPLAY_NAME "libOpenCV-dev")
SET(CPACK_COMPONENT_dev_SECTION "libdevel")
SET(CPACK_COMPONENT_docs_DEPENDS "libs")
SET(CPACK_COMPONENT_docs_DESCRIPTION "Documentation for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_docs_DISPLAY_NAME "libOpenCV-docs")
SET(CPACK_COMPONENT_docs_SECTION "doc")
SET(CPACK_COMPONENT_java_DEPENDS "libs")
SET(CPACK_COMPONENT_java_DESCRIPTION "Java bindings for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_java_DISPLAY_NAME "libOpenCV-java")
SET(CPACK_COMPONENT_java_SECTION "java")
SET(CPACK_COMPONENT_libs_DESCRIPTION "Open Computer Vision Library")
SET(CPACK_COMPONENT_libs_DISPLAY_NAME "libOpenCV")
SET(CPACK_COMPONENT_libs_SECTION "libs")
SET(CPACK_COMPONENT_python_DEPENDS "libs")
SET(CPACK_COMPONENT_python_DESCRIPTION "Python bindings for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_python_DISPLAY_NAME "libOpenCV-python")
SET(CPACK_COMPONENT_python_SECTION "python")
SET(CPACK_COMPONENT_samples_DEPENDS "libs")
SET(CPACK_COMPONENT_samples_DESCRIPTION "Samples for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_samples_DISPLAY_NAME "libOpenCV-samples")
SET(CPACK_COMPONENT_samples_SECTION "devel")
SET(CPACK_COMPONENT_tests_DEPENDS "libs")
SET(CPACK_COMPONENT_tests_DESCRIPTION "Accuracy and performance tests for Open Source Computer Vision Library")
SET(CPACK_COMPONENT_tests_DISPLAY_NAME "libOpenCV-tests")
SET(CPACK_COMPONENT_tests_SECTION "misc")
SET(CPACK_DEBIAN_ARCHITECTURE "amd64")
SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://opencv.org")
SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
SET(CPACK_DEBIAN_PACKAGE_SECTION "libs")
SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS "TRUE")
SET(CPACK_DEB_COMPONENT_INSTALL "TRUE")
SET(CPACK_DEB_python_PACKAGE_DEPENDS "python2.7")
SET(CPACK_GENERATOR "TGZ;TBZ2;TZ")
SET(CPACK_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp$;\\.#;/#")
SET(CPACK_INSTALLED_DIRECTORIES "/home/christian/git/OpenCV4Android/opencv-2.4.10;/")
SET(CPACK_INSTALL_CMAKE_PROJECTS "")
SET(CPACK_INSTALL_PREFIX "/home/christian/android_install")
SET(CPACK_MODULE_PATH "")
SET(CPACK_NSIS_DISPLAY_NAME "OpenCV dff53e3-dirty")
SET(CPACK_NSIS_INSTALLER_ICON_CODE "")
SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
SET(CPACK_NSIS_PACKAGE_NAME "OpenCV dff53e3-dirty")
SET(CPACK_OUTPUT_CONFIG_FILE "/home/christian/git/OpenCV4Android/opencv-2.4.10/build/CPackConfig.cmake")
SET(CPACK_PACKAGE_CONTACT "admin@opencv.org")
SET(CPACK_PACKAGE_DEFAULT_LOCATION "/")
SET(CPACK_PACKAGE_DESCRIPTION "OpenCV (Open Source Computer Vision Library) is an open source computer vision
and machine learning software library. OpenCV was built to provide a common
infrastructure for computer vision applications and to accelerate the use of
machine perception in the commercial products. Being a BSD-licensed product,
OpenCV makes it easy for businesses to utilize and modify the code.")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "/usr/share/cmake-2.8/Templates/CPack.GenericDescription.txt")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open Computer Vision Library")
SET(CPACK_PACKAGE_FILE_NAME "OpenCV-dff53e3-dirty-x86_64")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "OpenCV dff53e3-dirty")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "OpenCV dff53e3-dirty")
SET(CPACK_PACKAGE_NAME "OpenCV")
SET(CPACK_PACKAGE_RELOCATABLE "true")
SET(CPACK_PACKAGE_VENDOR "OpenCV Foundation")
SET(CPACK_PACKAGE_VERSION "dff53e3-dirty")
SET(CPACK_PACKAGE_VERSION_MAJOR "2")
SET(CPACK_PACKAGE_VERSION_MINOR "4")
SET(CPACK_PACKAGE_VERSION_PATCH "10")
SET(CPACK_RESOURCE_FILE_LICENSE "/home/christian/git/OpenCV4Android/opencv-2.4.10/LICENSE")
SET(CPACK_RESOURCE_FILE_README "/usr/share/cmake-2.8/Templates/CPack.GenericDescription.txt")
SET(CPACK_RESOURCE_FILE_WELCOME "/usr/share/cmake-2.8/Templates/CPack.GenericWelcome.txt")
SET(CPACK_RPM_COMPONENT_INSTALL "TRUE")
SET(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
SET(CPACK_RPM_PACKAGE_DESCRIPTION "OpenCV (Open Source Computer Vision Library) is an open source computer vision
and machine learning software library. OpenCV was built to provide a common
infrastructure for computer vision applications and to accelerate the use of
machine perception in the commercial products. Being a BSD-licensed product,
OpenCV makes it easy for businesses to utilize and modify the code.")
SET(CPACK_RPM_PACKAGE_LICENSE "BSD")
SET(CPACK_RPM_PACKAGE_SUMMARY "Open Computer Vision Library")
SET(CPACK_RPM_PACKAGE_URL "http://opencv.org")
SET(CPACK_SET_DESTDIR "OFF")
SET(CPACK_SOURCE_CYGWIN "")
SET(CPACK_SOURCE_GENERATOR "TGZ;TBZ2;TZ")
SET(CPACK_SOURCE_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp$;\\.#;/#")
SET(CPACK_SOURCE_INSTALLED_DIRECTORIES "/home/christian/git/OpenCV4Android/opencv-2.4.10;/")
SET(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/christian/git/OpenCV4Android/opencv-2.4.10/build/CPackSourceConfig.cmake")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "OpenCV-dff53e3-dirty-x86_64")
SET(CPACK_SOURCE_TBZ2 "ON")
SET(CPACK_SOURCE_TGZ "ON")
SET(CPACK_SOURCE_TOPLEVEL_TAG "Linux-Source")
SET(CPACK_SOURCE_TZ "ON")
SET(CPACK_SOURCE_ZIP "OFF")
SET(CPACK_STRIP_FILES "")
SET(CPACK_SYSTEM_NAME "Linux")
SET(CPACK_TOPLEVEL_TAG "Linux-Source")
SET(CPACK_dev_COMPONENT_INSTALL "TRUE")
SET(CPACK_docs_COMPONENT_INSTALL "TRUE")
SET(CPACK_java_COMPONENT_INSTALL "TRUE")
SET(CPACK_libs_COMPONENT_INSTALL "TRUE")
SET(CPACK_python_COMPONENT_INSTALL "TRUE")
SET(CPACK_samples_COMPONENT_INSTALL "TRUE")
SET(CPACK_set_DESTDIR "on")
