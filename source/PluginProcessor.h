#pragma once

#include <JuceHeader.h>
#include <anira/anira.h>
#include "GANSynthInference.h"
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


    juce::MidiMessageCollector& getMidiMessageCollector() noexcept { return midiMessageCollector; }

    bool isGenerating() const { return m_isGenerating; }

    void generateAudio();

    const juce::AudioSampleBuffer& getGeneratedAudio() const { return m_generatedAudio; }

private:
    float nextGaussian(juce::Random& r);

    void parameterChanged (const juce::String& parameterID, float newValue) override;
    // Offline inference and playback
    GANSynthInference m_inference;
    juce::AudioSampleBuffer m_generatedAudio;
    std::atomic<bool> m_isGenerating {false};
    std::atomic<int> m_playIndex {0};
    std::atomic<bool> m_isPlaying {false};
    int m_lastMidiNote = 60;
  
    juce::AudioProcessorValueTreeState apvts;

    juce::dsp::Gain<float> gain;
    float gainParam;

    juce::MidiMessageCollector midiMessageCollector;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GANSynth_for_MIDISynthesizer_Processor)
};
