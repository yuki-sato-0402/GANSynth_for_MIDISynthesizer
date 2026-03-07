#include "CustomBackendProcessor.h"
#include <anira/utils/Logger.h>
#include <algorithm>

namespace anira {

CustomBackend::CustomBackend(InferenceConfig& inference_config) : BackendBase(inference_config)
{
    // Initialize instances based on the number of parallel processors specified in the inference configuration
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

CustomBackend::~CustomBackend() {
}

void CustomBackend::prepare() {
    std::cout << "$[CustomBackend] Preparing instance for inference...$" << std::endl;
    for(auto& instance : m_instances) {
        instance->prepare();
    }
}

void CustomBackend::process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) {
    // Find an available instance and run inference
    for(auto& instance : m_instances) {
        if (!(instance->m_processing.exchange(true))) {
            instance->process(input, output, session);
            instance->m_processing.exchange(false);
            return;
        }
    }
    // Fallback if all instances are busy
    m_instances[0]->process(input, output, session);
}

CustomBackend::Instance::Instance(InferenceConfig& inference_config) : m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
                                                                    m_inference_config(inference_config)
{
    std::cout << "$[CustomBackend] Initializing ONNX Runtime session...$" << std::endl;
    m_session_options.SetIntraOpNumThreads(1);

    // Check if the model is binary
    if (m_inference_config.is_model_binary(anira::InferenceBackend::CUSTOM)) {
        const anira::ModelData* model_data = m_inference_config.get_model_data(anira::InferenceBackend::CUSTOM);
        assert(model_data && "Model data not found for binary model!");

        // Load model from binary data
        m_session = std::make_unique<Ort::Session>(m_env, model_data->m_data, model_data->m_size, m_session_options);
    } else {
        // Load model from file path
#ifdef _WIN32
        std::string modelpath_str = m_inference_config.get_model_path(anira::InferenceBackend::CUSTOM);
        std::wstring modelpath = std::wstring(modelpath_str.begin(), modelpath_str.end());
#else
        std::string modelpath = m_inference_config.get_model_path(anira::InferenceBackend::CUSTOM);
#endif
        m_session = std::make_unique<Ort::Session>(m_env, modelpath.c_str(), m_session_options);
    }
    
    m_input_names.resize(m_session->GetInputCount());
    m_output_names.resize(m_session->GetOutputCount());
    m_input_name.clear();
    m_output_name.clear();

    for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
        m_input_name.emplace_back(m_session->GetInputNameAllocated(i, m_ort_alloc));
        m_input_names[i] = m_input_name[i].get();
    }
    for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
        m_output_name.emplace_back(m_session->GetOutputNameAllocated(i, m_ort_alloc));
        m_output_names[i] = m_output_name[i].get();
    }

    // Allocate persistent memory for input tensors
    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());

    m_inputs.clear();
    // The first input is of type int32.
    m_note_number_data.resize(m_inference_config.get_tensor_input_size()[0]);

    //std::cout << "m_inference_config.get_tensor_input_size()[0]: " << m_inference_config.get_tensor_input_size()[0] << std::endl;
    //std::cout << "m_inference_config.get_tensor_input_size()[1]: " << m_inference_config.get_tensor_input_size()[1] << std::endl;
    m_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
        m_memory_info,
        m_note_number_data.data(),
        m_note_number_data.size(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[0].data(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[0].size()
    ));

    // The second input is of type float.
    m_input_data[0].resize(m_inference_config.get_tensor_input_size()[1]);
    m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        m_memory_info,
        m_input_data[0].data(),
        m_input_data[0].size(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[1].data(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[1].size()
    ));

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        try {
            m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
        } catch (Ort::Exception &e) {
            LOG_ERROR << "[CustomBackend] Warm-up Error: " << e.what() << std::endl;
        }
    }

    std::cout << "$[CustomBackend] ONNX Runtime session initialized successfully!$" << std::endl;
}

CustomBackend::Instance::~Instance() {
    m_session.reset();
}
    
void CustomBackend::Instance::prepare() {
    std::cout << "$[CustomBackend] Preparing instance for inference...$" << std::endl;
    for (auto & i : m_input_data) {
        i.clear();
    }
    for (auto & i : m_note_number_data) {
        i = 0;
    }

    std::cout << "$[CustomBackend] Instance prepared successfully!$" << std::endl;

    
}

void CustomBackend::Instance::process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) {
    (void)session;  

    // Input 0: int32 type (Note Number [1])
    m_note_number_data[0] =  static_cast<int32_t>(input[0].data()[0]);
    m_inputs[0] = Ort::Value::CreateTensor<int32_t>(
        m_memory_info,
        m_note_number_data.data(),
        1,
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[0].data(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[0].size()
    );

    // Input 1: float type (noise vector [1, 256])
    m_inputs[1] = Ort::Value::CreateTensor<float>(
        m_memory_info,
        input[1].data(),
        input[1].get_num_samples() * input[1].get_num_channels(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[1].data(),
        m_inference_config.get_tensor_input_shape(anira::InferenceBackend::CUSTOM)[1].size()
    );

    // Run inference
    try {
        m_outputs = m_session->Run(Ort::RunOptions{nullptr}, 
                                   m_input_names.data(), 
                                   m_inputs.data(), 
                                   m_inputs.size(), 
                                   m_output_names.data(), 
                                   m_output_names.size());
    } catch (Ort::Exception &e) {
        LOG_ERROR << "[CustomBackend] ONNX Runtime Error: " << e.what() << std::endl;
    }

    // Output: [1, 128, 1024, 2]
    const auto output_read_ptr = m_outputs[0].GetTensorMutableData<float>();
    size_t out_size = m_inference_config.get_tensor_output_size()[0];
    for (size_t j = 0; j < out_size; j++) {
        output[0].data()[j] = output_read_ptr[j];
    }
}

} // namespace anira
