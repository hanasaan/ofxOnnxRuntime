meta:
	ADDON_NAME = ofxOnnxRuntime
	ADDON_DESCRIPTION = "ONNX Runtime addon for OpenFrameworks"
	ADDON_AUTHOR = Yuya Hanai
	ADDON_TAGS = "ONNX"
	ADDON_URL = https://github.com/hanasaan/ofxOnnxRuntime

common:
	ADDON_INCLUDES = libs/onnxruntime/include
	ADDON_INCLUDES += src
osx:
	ADDON_LDFLAGS = -Xlinker -rpath -Xlinker @executable_path
vs:

