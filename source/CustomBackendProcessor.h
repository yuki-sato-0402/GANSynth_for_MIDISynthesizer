#ifndef ANIRA_CUSTOMBACKEND_H
#define ANIRA_CUSTOMBACKEND_H


#include <JuceHeader.h>
#include "../modules/anira/include/anira/InferenceConfig.h"
#include "../modules/anira/include/anira/utils/Buffer.h"
#include "../modules/anira/include/anira/backends/BackendBase.h"
#include "../modules/anira/include/anira/scheduler/SessionElement.h"

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace anira {

/**
 * @brief Custom ONNX Runtime-based neural network inference processor (Pass-through)
 */
class ANIRA_API CustomBackend : public BackendBase {
public:
    CustomBackend(InferenceConfig& inference_config);
    ~CustomBackend() override;

    void prepare() override;
    void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) override;

private:
    struct Instance {
        Instance(InferenceConfig& inference_config);
        ~Instance();

        void prepare();
        void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session);

        Ort::MemoryInfo m_memory_info;
        Ort::Env m_env;
        Ort::AllocatorWithDefaultOptions m_ort_alloc;
        Ort::SessionOptions m_session_options;

        std::unique_ptr<Ort::Session> m_session;

        std::vector<MemoryBlock<float>> m_input_data;
        //std::vector<MemoryBlock<int32_t>> m_note_number_data;
        std::vector<int32_t> m_note_number_data;

        std::vector<Ort::Value> m_inputs;
        std::vector<Ort::Value> m_outputs;

        // Input array
        std::vector<Ort::Value> inputs;

        std::vector<Ort::AllocatedStringPtr> m_input_name;
        std::vector<Ort::AllocatedStringPtr> m_output_name;

        std::vector<const char *> m_output_names;
        std::vector<const char *> m_input_names;

        InferenceConfig& m_inference_config;
        std::atomic<bool> m_processing {false};

#if DOXYGEN
        // Since Doxygen does not find classes structures nested in std::shared_ptr
        MemoryBlock<float>* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
    };

    std::vector<std::shared_ptr<Instance>> m_instances;

#if DOXYGEN
    Instance* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif //ANIRA_CUSTOMBACKEND_H
