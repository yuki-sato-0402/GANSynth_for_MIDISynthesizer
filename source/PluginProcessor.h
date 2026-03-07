#pragma once

#include <JuceHeader.h>
#include <anira/anira.h>
#include "GANSynthModelConfig.h"
#include "CustomBackendProcessor.h"
//==============================================================================
class GANSynth_for_MIDISynthesizer_Processor  : public juce::AudioProcessor, public juce::AudioProcessorValueTreeState::Listener, 
public juce::ValueTree::Listener, public juce::ActionBroadcaster
{
public:
    //==============================================================================
    GANSynth_for_MIDISynthesizer_Processor();
    ~GANSynth_for_MIDISynthesizer_Processor() override = default;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //juce::AudioProcessorValueTreeState& getValueTreeState() { return apvts; }

    juce::MidiMessageCollector& getMidiMessageCollector() noexcept { return midiMessageCollector; }

    void triggerInference(int midiNote);
    bool isGenerating() const { return m_isGenerating; }

private:
    void parameterChanged (const juce::String& parameterID, float newValue) override;

  
    juce::AudioProcessorValueTreeState apvts;
    juce::ValueTree valueTree;

    juce::dsp::Gain<float> gain;
    float gainParam;

    // Optional ContextConfig
    //anira::ContextConfig anira_context_config;

    anira::InferenceConfig inference_config = GANSynth_model_config;
    anira::PrePostProcessor pp_processor;
    anira::CustomBackend custom_backend;
    anira::InferenceHandler inference_handler;

    // Offline inference and playback
    juce::AudioSampleBuffer m_generatedAudio;
    std::atomic<bool> m_isGenerating {false};
    std::atomic<int> m_playIndex {0};
    std::atomic<bool> m_isPlaying {false};
    int m_lastMidiNote = 60;

    double mutedSamples = 0;
    int64_t totalSamplesProcessed = 0;

    juce::MidiMessageCollector midiMessageCollector;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GANSynth_for_MIDISynthesizer_Processor)
};
