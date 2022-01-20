#include "ofMain.h"

#include "ofxOnnxRuntime.h"

// code from
// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/MNIST/MNIST.cpp
template <typename T> static void softmax(T &input) {
	float rowmax = *std::max_element(input.begin(), input.end());
	std::vector<float> y(input.size());
	float sum = 0.0f;
	for (size_t i = 0; i != input.size(); ++i) {
		sum += y[i] = std::exp(input[i] - rowmax);
	}
	for (size_t i = 0; i != input.size(); ++i) {
		input[i] = y[i] / sum;
	}
}

class ofApp : public ofBaseApp {
	ofxOnnxRuntime::BaseHandler mnist2;
	vector<float> mnist_result;

	ofFbo fbo_render;
	ofFbo fbo_classification;
	ofFloatPixels pix;
	bool prev_pressed = false;
	glm::vec2 prev_pt;

public:
	void setup() {
		ofSetVerticalSync(true);
		ofSetFrameRate(60);

#ifdef _MSC_VER
		mnist2.setup("mnist-8.onnx", ofxOnnxRuntime::BaseSetting{ ofxOnnxRuntime::INFER_TENSORRT });
#else
		mnist2.setup("mnist-8.onnx");
#endif
		fbo_render.allocate(280, 280, GL_RGB, 0);
		fbo_render.getTexture().setTextureMinMagFilter(GL_NEAREST, GL_NEAREST);
		fbo_render.begin();
		ofClear(0);
		fbo_render.end();
		fbo_classification.allocate(28, 28, GL_R32F, 0);

		//pix.setFromExternalPixels(&mnist->input_image_.front(), 28, 28, 1);
		pix.setFromExternalPixels(mnist2.getInputTensorData(), 28, 28, 1);

		//mnist->Run();
        auto& result = mnist2.run();
        const float *output_ptr = result.GetTensorMutableData<float>();

		mnist_result.resize(10);
        
        cerr << "API : " << Ort::Global<void>::api_ << endl;
	}

	void update() {
		if (ofGetMousePressed()) {
			auto pt = glm::vec2(ofGetMouseX(), ofGetMouseY() - 60);
			fbo_render.begin();
			ofPushStyle();
			ofSetColor(255);
			if (prev_pressed) {
				ofSetLineWidth(20);
				ofDrawLine(prev_pt, pt);
			}
			ofDrawCircle(pt.x, pt.y, 10);
			ofPopStyle();
			fbo_render.end();

			fbo_classification.begin();
			ofClear(0);
			fbo_render.draw(0, 0, fbo_classification.getWidth(),
				fbo_classification.getHeight());
			fbo_classification.end();
			fbo_classification.readToPixels(pix);
			auto& result = mnist2.run();
			const float *output_ptr = result.GetTensorMutableData<float>();
			memcpy(mnist_result.data(), output_ptr, mnist_result.size() * sizeof(float));
			softmax(mnist_result);
			prev_pt = pt;
			prev_pressed = true;
		}
		else {
			prev_pressed = false;
		}
	}

	void draw() {
		ofClear(128);

		fbo_render.draw(0, 60);
		fbo_classification.draw(0, 340);

		// render result
		auto& result = mnist_result;
		for (int i = 0; i < 10; ++i) {
			stringstream ss;
			ss << i << ":" << std::fixed << std::setprecision(3)
				<< mnist_result[i];
			ofDrawBitmapString(ss.str(), 300, 70 + i * 30);
			ofPushStyle();
			ofSetColor(0, 255, 0);
			ofDrawRectangle(360.0, 55 + i * 30, mnist_result[i] * 300.0, 20);
			ofPopStyle();
		}

		stringstream ss;
		ss << "FPS : " << ofGetFrameRate() << endl;
		ss << "Draw any digit (0-9) here" << endl;
		ss << "Press c to clear buffer";
		ofDrawBitmapStringHighlight(ss.str(), 10, 20);
	}

	void keyPressed(int key) {
		if (key == 'c') {
			fbo_render.begin();
			ofClear(0);
			fbo_render.end();
		}
	}

	void keyReleased(int key) {}
	void mouseMoved(int x, int y) {}
	void mouseDragged(int x, int y, int button) {}

	void mousePressed(int x, int y, int button) {}

	void mouseReleased(int x, int y, int button) {}
	void windowResized(int w, int h) {}
	void dragEvent(ofDragInfo dragInfo) {}
	void gotMessage(ofMessage msg) {}
};

//========================================================================
int main() {
	ofSetupOpenGL(640, 400, OF_WINDOW); // <-------- setup the GL context

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());
}
