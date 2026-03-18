#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
//#include <anira/anira.h>

class GANSynthInference {

/*
https://github.com/magenta/magenta/blob/main/magenta/models/gansynth/lib/specgrams_helper.py was referenced.
1.  Logarithm cancellation: Calculate exp(logmelmag2) to return to Mel-scale power.
2.  Mel-to-Linear Restoration:
    Stretch the Mel-scale data to a linear frequency axis (1025 bins) using a pseudo-inverse matrix.
3.  Phase Restoration: Take the cumulative sum (cumsum) of instantaneous frequencies to return to phase.
4.  ISTFT (Inverse Short-Time Fourier Transform):
    Combine the amplitude and phase to return to a complex number, 
    and then synthesize a time-domain audio waveform using an inverse Fourier transform.
*/
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
    // For each linear bin i, stores which mel bins j have non-zero weight
    std::vector<std::vector<SparseElement>> m_sparseMel2l;

    Ort::Env m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_memoryInfo;
    Ort::SessionOptions m_sessionOptions;

    // Mel-to-Linear matrix (1024 x 1025)
    // std::vector<float> m_mel2l; // No longer needed as dense
    bool m_mel2lLoaded = false; 

    // Constants for GANSynth
    const int m_sampleRate = 16000;
    double m_targetSampleRate = 0;

    const int m_nFFT = 2048;
    const int m_hopLength = 512;
    const int m_nMel = 1024;
    const int m_nMag = 1025;
    const int m_numFrames = 128;

    std::unique_ptr<juce::dsp::FFT> m_fft;
    
    // Pre-calculated window and normalization
    std::vector<float> m_window;
    std::vector<float> m_olaNormalization;
    
    juce::LagrangeInterpolator resampler;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GANSynthInference)
};
