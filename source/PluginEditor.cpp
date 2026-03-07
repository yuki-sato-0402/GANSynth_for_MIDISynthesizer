/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/
#include "PluginEditor.h"
#include "PluginProcessor.h"

GANSynth_for_MIDISynthesizer_ProcessorEditor::GANSynth_for_MIDISynthesizer_ProcessorEditor(GANSynth_for_MIDISynthesizer_Processor& p, juce::AudioProcessorValueTreeState& apvts)
    : AudioProcessorEditor (&p),processorRef (p),  valueTreeState(apvts)
{
  GainLabel.setText ("Output Gain", juce::dontSendNotification);
  GainLabel.setJustificationType(juce::Justification::centred);
  GainLabel.setColour(juce::Label::textColourId, juce::Colours::black);
  addAndMakeVisible(GainLabel);

  GainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
  GainSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  addAndMakeVisible(GainSlider);
  GainSliderAttachment = std::make_unique<SliderAttachment>(valueTreeState, "outputGain", GainSlider);

  addAndMakeVisible(generateButton);
  generateButton.onClick = [this] {
    processorRef.triggerInference(60); // Default to C4 for now
  };

  addAndMakeVisible(midiKeyboardComponent);
  midiKeyboardState.addListener(&processorRef.getMidiMessageCollector());

  startTimer(100);
  setSize(920, 520);
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::paint (juce::Graphics& g)
{
  g.fillAll(juce::Colours::darkgrey);
  
  if (processorRef.isGenerating())
  {
      g.setColour(juce::Colours::white);
      g.drawText("Generating...", getLocalBounds(), juce::Justification::centred);
  }
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::resized()
{
  auto area = getLocalBounds().reduced(20);
  
  auto topArea = area.removeFromTop(200);
  GainSlider.setBounds(topArea.removeFromLeft(150));
  GainLabel.setBounds(GainSlider.getX(), GainSlider.getY() - 20, GainSlider.getWidth(), 20);
  
  generateButton.setBounds(area.removeFromTop(50).reduced(100, 0));
  
  midiKeyboardComponent.setBounds(area.removeFromBottom(100));
}



