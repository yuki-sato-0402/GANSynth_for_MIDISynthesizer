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

class GANSynth_for_MIDISynthesizer_ProcessorEditor  : public juce::AudioProcessorEditor, public juce::Timer
{
public:
  GANSynth_for_MIDISynthesizer_ProcessorEditor(GANSynth_for_MIDISynthesizer_Processor& p, juce::AudioProcessorValueTreeState& apvts);
  ~GANSynth_for_MIDISynthesizer_ProcessorEditor() override = default;

  //==============================================================================
  void paint (juce::Graphics& g) override;
  void resized() override; 
  void timerCallback() override { repaint(); }
  typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;
  typedef juce::AudioProcessorValueTreeState::ComboBoxAttachment ComboBoxAttachment;

private:
  GANSynth_for_MIDISynthesizer_Processor& processorRef;
  juce::AudioProcessorValueTreeState& valueTreeState;

  juce::Slider GainSlider;
  
  juce::Label  GainLabel;
  
  juce::TextButton generateButton { "Generate" };


  std::unique_ptr<SliderAttachment> GainSliderAttachment;
 
  juce::MidiKeyboardState midiKeyboardState;
  juce::MidiKeyboardComponent midiKeyboardComponent { midiKeyboardState, juce::MidiKeyboardComponent::horizontalKeyboard };

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GANSynth_for_MIDISynthesizer_ProcessorEditor)
};
#endif //NN_INFERENCE_TEMPLATE_PLUGINPARAMETERS_H
