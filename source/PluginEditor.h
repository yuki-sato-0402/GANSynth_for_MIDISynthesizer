#ifndef NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
#define NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

class GANSynth_for_MIDISynthesizer_ProcessorEditor  : public juce::AudioProcessorEditor, 
                                            public juce::Timer,
                                            public juce::ActionListener
{
public:
  GANSynth_for_MIDISynthesizer_ProcessorEditor(GANSynth_for_MIDISynthesizer_Processor& p, juce::AudioProcessorValueTreeState& apvts);
  ~GANSynth_for_MIDISynthesizer_ProcessorEditor() override;

  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override; 
  void timerCallback() override { repaint(); }
  void actionListenerCallback (const juce::String& message) override;

  typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;
  typedef juce::AudioProcessorValueTreeState::ComboBoxAttachment ComboBoxAttachment;

private:
  struct WaveformComponent : public juce::Component
  {
      void paint (juce::Graphics& g) override;
      void setThumbnail (const juce::AudioSampleBuffer& buffer);
      juce::AudioSampleBuffer waveformBuffer;
  };

  GANSynth_for_MIDISynthesizer_Processor& processorRef;
  juce::AudioProcessorValueTreeState& valueTreeState;

  juce::Slider GainSlider;
  juce::Label  GainLabel;
  juce::TextButton generateButton { "Generate" };

  WaveformComponent waveformComponent;

  std::unique_ptr<SliderAttachment> GainSliderAttachment;
 
  juce::MidiKeyboardState midiKeyboardState;
  juce::MidiKeyboardComponent midiKeyboardComponent { midiKeyboardState, juce::MidiKeyboardComponent::horizontalKeyboard };

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GANSynth_for_MIDISynthesizer_ProcessorEditor)
};
#endif //NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
