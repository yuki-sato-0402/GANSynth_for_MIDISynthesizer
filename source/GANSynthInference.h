#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

class GANSynthInference {
public:
    GANSynthInference();
    ~GANSynthInference();

    void prepare(const juce::String& modelPath);
    void setTargetSampleRate(double sampleRate) { m_targetSampleRate = sampleRate; }
    bool loadMel2lFromCsv(const juce::String& csvPath);
    void generate(int midiNote, const std::vector<float>& latentVector, juce::AudioBuffer<float>& outputBuffer);

private:
    void postProcess(const float* rawOutput, juce::AudioBuffer<float>& outputBuffer);

    struct SparseElement {
        int index;
        float weight;
    };
    std::vector<std::vector<SparseElement>> m_sparseMel2l;

    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memoryInfo;
    Ort::SessionOptions m_sessionOptions;

    bool m_mel2lLoaded = false; 

    const int m_sampleRate = 16000;
    double m_targetSampleRate = 0;

    const int m_nFFT = 2048;
    const int m_hopLength = 512;
    const int m_nMel = 1024;
    const int m_nMag = 1025;
    const int m_numFrames = 128;

    std::unique_ptr<juce::dsp::FFT> m_fft;
    
    std::vector<float> m_window;
    std::vector<float> m_olaNormalization;
    
    juce::LagrangeInterpolator resampler;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GANSynthInference)
};
