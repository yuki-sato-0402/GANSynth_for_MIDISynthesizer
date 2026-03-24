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

    const juce::AudioSampleBuffer& getGeneratedAudio() const { 
        // If no notes have been generated, return an empty buffer to avoid issues in the editor
        if (m_generatedNotes.empty()) return m_emptyBuffer;
        // Find the closest generated note to the last MIDI note played
        auto it = m_generatedNotes.find(m_lastMidiNote);
        if (it != m_generatedNotes.end()) return it->second;
        return m_generatedNotes.begin()->second;
    }

private:
    float nextGaussian(juce::Random& r);

    void parameterChanged (const juce::String& parameterID, float newValue) override;
    
    // Voice structure for polyphony
    struct Voice {
        int noteNumber = -1;
        int baseNote = -1; // Added: Store the closest pre-generated note
        int playIndex = 0;
        double pitchRatio = 1.0;
        juce::LagrangeInterpolator interpolator;
        bool active = false;

        void reset() {
            noteNumber = -1;
            baseNote = -1;
            playIndex = 0;
            pitchRatio = 1.0;
            interpolator.reset();
            active = false;
        }
    };
    std::array<Voice, 6> m_voices;

    // Offline inference and playback
    GANSynthInference m_inference;
    // Cache for generated audio samples for MIDI notes
    std::map<int, juce::AudioSampleBuffer> m_generatedNotes;
    juce::AudioSampleBuffer m_emptyBuffer;

    std::atomic<bool> m_isGenerating {false};
    int m_lastMidiNote = 60;

    juce::AudioProcessorValueTreeState apvts;


    juce::dsp::Gain<float> gain;
    float gainParam;

    juce::MidiMessageCollector midiMessageCollector;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GANSynth_for_MIDISynthesizer_Processor)
};
