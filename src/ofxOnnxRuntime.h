#pragma once

#include <onnxruntime_cxx_api.h>

namespace ofxOnnxRuntime
{
	enum InferType
	{
		INFER_CPU = 0,
		INFER_CUDA,
		INFER_TENSORRT
	};

	struct BaseSetting
	{
		InferType infer_type;
		int device_id;
	};

	class BaseHandler
	{
	public:
		void setup(const std::string& onnx_path, const BaseSetting& base_setting = BaseSetting{ INFER_CPU, 0 });
		void setup2(const std::string& onnx_path, const Ort::SessionOptions& session_options);

		Ort::Value& run();

		float* getInputTensorData() {
			return this->input_values_handler.data();
		}
	protected:
		Ort::Env ort_env;
		std::shared_ptr<Ort::Session> ort_session;
		std::vector<const char *> input_node_names;
		std::vector<int64_t> input_node_dims; // 1 input only.
		std::size_t input_tensor_size = 1;
		std::vector<float> input_values_handler;
		Ort::Value dummy_tensor_{ nullptr };
		Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::vector<const char *> output_node_names;
		std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
		int num_outputs = 1;
	};
}