#include "ofMain.h"

#include "ofxOnnxRuntime.h"

// code from
//https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/MNIST/MNIST.cpp
template <typename T>
static void softmax(T& input) {
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

// This is the structure to interface with the MNIST model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct MNIST {
  MNIST() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  }

  std::ptrdiff_t Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

 private:
  Ort::Env env;
  Ort::Session session_{env, ofToDataPath("mnist-8.onnx", true).c_str(), Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};

class ofApp : public ofBaseApp{
    shared_ptr<MNIST> mnist;
    ofFbo fbo_render;
    ofFbo fbo_classification;
    ofFloatPixels pix;
    bool prev_pressed = false;
    glm::vec2 prev_pt;
public:
    void setup()
    {
        ofSetVerticalSync(true);
        ofSetFrameRate(60);
        
        mnist = make_shared<MNIST>();
        
        fbo_render.allocate(280, 280, GL_RGB, 0);
        fbo_render.getTexture().setTextureMinMagFilter(GL_NEAREST, GL_NEAREST);
        fbo_render.begin();
        ofClear(0);
        fbo_render.end();
        fbo_classification.allocate(28, 28, GL_R32F, 0);
        
        pix.setFromExternalPixels(&mnist->input_image_.front(), 28, 28, 1);
    }
    
    void update()
    {
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
            fbo_render.draw(0, 0, fbo_classification.getWidth(), fbo_classification.getHeight());
            fbo_classification.end();
            fbo_classification.readToPixels(pix);
            mnist->Run();
            prev_pt = pt;
            prev_pressed = true;
        } else {
            prev_pressed = false;
        }
    }
    
    void draw()
    {
        ofClear(128);
        
        fbo_render.draw(0, 60);
        fbo_classification.draw(0, 340);
        
        // render result
        for (int i=0; i<10; ++i) {
            stringstream ss;
            ss << i << ":" << std::fixed << std::setprecision(3) << mnist->results_[i];
            ofDrawBitmapString(ss.str(), 300, 70 + i * 30);
            ofPushStyle();
            ofSetColor(0, 255, 0);
            ofDrawRectangle(360.0, 55 + i * 30, mnist->results_[i] * 300.0, 20);
            ofPopStyle();
        }
        
        stringstream ss;
        ss << "FPS : " << ofGetFrameRate() << endl;
        ss << "Draw any digit (0-9) here" << endl;
        ss << "Press c to clear buffer";
        ofDrawBitmapStringHighlight(ss.str(), 10, 20);
    }
    
    void keyPressed(int key)
    {
        if (key == 'c') {
            fbo_render.begin();
            ofClear(0);
            fbo_render.end();
        }
    }
    
    void keyReleased(int key) {}
    void mouseMoved(int x, int y ) {}
    void mouseDragged(int x, int y, int button) {}

    void mousePressed(int x, int y, int button) {
        
    }
    
    void mouseReleased(int x, int y, int button) {}
    void windowResized(int w, int h) {}
    void dragEvent(ofDragInfo dragInfo) {}
    void gotMessage(ofMessage msg) {}
    
};

//========================================================================
int main( ){
    ofSetupOpenGL(640,400,OF_WINDOW);            // <-------- setup the GL context
    
    // this kicks off the running of my app
    // can be OF_WINDOW or OF_FULLSCREEN
    // pass in width and height too:
    ofRunApp(new ofApp());
    
}
