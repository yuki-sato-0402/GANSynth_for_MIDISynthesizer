#ifndef ANIRA_GANSynth_model_CONFIG_H
#define ANIRA_GANSynth_model_CONFIG_H

#include <anira/anira.h>

static std::vector<anira::ModelData> modelData = {
    {GANSynth_MODEL_DIR + std::string("gansynth.onnx"), anira::InferenceBackend::CUSTOM},
};

static std::vector<anira::TensorShape> tensorShape = {
    {
        {{1}, {1, 256}}, // List of input tensor shapes, where each shape is a vector of dimensions
        {{1, 128, 1024, 2}}, // List of output tensor shapes, where each shape is a vector of dimensions
        anira::InferenceBackend::CUSTOM
    },
};

static anira::ProcessingSpec processingSpec{
    {1, 1},  // preprocess_input_channels – Number of input channels for each input tensor
    {1},     // preprocess_output_channels – Number of output channels for each output tensor
    {0, 0},// preprocess_input_size – Samples count required for preprocessing for each input tensor (0 = non-streamable)
    {0}, // postprocess_output_size – Samples count after the postprocessing for each output tensor (0 = non-streamable)
    {262144},   // internal_model_latency – Internal model latency in samples for each output tensor
};

static anira::InferenceConfig GANSynth_model_config(
    modelData,
    tensorShape,
    processingSpec,
    1000.00f,   //  Maximum allowed inference time in milliseconds per inference
    2,        // Number of warm-up inferences to perform during initialization
    false,    // Whether to use exclusive processor sessions
    0.0f     // Ratio controlling blocking behavior (0.0-1.0)
);

#endif //ANIRA_GANSynth_model_CONFIG_H
