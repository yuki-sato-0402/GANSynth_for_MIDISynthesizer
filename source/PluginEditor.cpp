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
  processorRef.addActionListener(this);

  GainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
  GainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, GainSlider.getTextBoxWidth(), GainSlider.getTextBoxHeight());
  addAndMakeVisible(GainSlider);
  GainSliderAttachment.reset (new SliderAttachment (valueTreeState, "outputGain", GainSlider));
  GainSlider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colours::white);
  GainSlider.setColour(juce::Slider::rotarySliderFillColourId, juce::Colours::midnightblue.brighter(0.25).withAlpha(0.75f));
  GainSlider.setColour(juce::Slider::thumbColourId , juce::Colours::midnightblue.brighter(0.1f));
  GainSlider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
  GainSlider.setColour(juce::Slider::textBoxOutlineColourId , juce::Colours::midnightblue.brighter(0.25));

  addAndMakeVisible(GainLabel);
  GainLabel.setText ("Output Gain", juce::dontSendNotification);
  GainLabel.setJustificationType(juce::Justification::centred);
  GainLabel.setColour(juce::Label::textColourId, juce::Colours::white);
  

  addAndMakeVisible(generateButton);
  generateButton.onClick = [this] {
    processorRef.generateAudio();
  };
  generateButton.setColour(juce::TextButton::buttonColourId, juce::Colours::midnightblue.brighter(0.1f));
  generateButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white);


  addAndMakeVisible(midiKeyboardComponent);
  midiKeyboardState.addListener(&processorRef.getMidiMessageCollector());
  midiKeyboardComponent.setAvailableRange (24, 84);

  addAndMakeVisible(waveformComponent);

  startTimer(100);
  setSize(620, 320);
}

GANSynth_for_MIDISynthesizer_ProcessorEditor::~GANSynth_for_MIDISynthesizer_ProcessorEditor()
{
  processorRef.removeActionListener(this);
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::actionListenerCallback(const juce::String& message)
{
  if (message == "GenerationFinished")
  {
      waveformComponent.setThumbnail(processorRef.getGeneratedAudio());
  }
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::WaveformComponent::paint(juce::Graphics& g)
{
  g.fillAll(juce::Colours::black.withAlpha(0.2f));
  g.setColour(juce::Colours::white);
  
  if (waveformBuffer.getNumSamples() > 0)
  {
      auto area = getLocalBounds().toFloat();
      auto width = area.getWidth();
      auto height = area.getHeight();
      auto midY = height / 2.0f;
      
      auto* reader = waveformBuffer.getReadPointer(0);
      int numSamples = waveformBuffer.getNumSamples();
      
      int samplesPerPixel = std::max(1, numSamples / (int)width);
      
      juce::Path p;
      p.startNewSubPath(0, midY);
      
      for (int x = 0; x < (int)width; ++x)
      {
          int sampleIdx = x * samplesPerPixel;
          if (sampleIdx >= numSamples) break;
          
          float min = 0;
          float max = 0;
          for (int i = 0; i < samplesPerPixel && (sampleIdx + i) < numSamples; ++i)
          {
              float val = reader[sampleIdx + i];
              if (val < min) min = val;
              if (val > max) max = val;
          }
          
          p.lineTo((float)x, midY - max * midY);
          p.lineTo((float)x, midY - min * midY);
      }
      
      g.strokePath(p, juce::PathStrokeType(1.0f));
  }
  else
  {
      g.setColour(juce::Colours::white.withAlpha(0.5f));
      g.drawText("No audio generated", getLocalBounds(), juce::Justification::centred);
  }
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::WaveformComponent::setThumbnail(const juce::AudioSampleBuffer& buffer)
{
  waveformBuffer.makeCopyOf(buffer);
  repaint();
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::paint (juce::Graphics& g)
{
  g.fillAll(juce::Colours::midnightblue);
  
  if (processorRef.isGenerating())
  {
      g.setColour(juce::Colours::white);
      g.drawText("Generating...", getLocalBounds(), juce::Justification::centred);
  }
}

void GANSynth_for_MIDISynthesizer_ProcessorEditor::resized()
{
  auto area = getLocalBounds();
  auto componentWidth = (area.getWidth() - 60)/5;
  auto componentHeight = (area.getHeight() - 80)/3;
  auto padding = 20;        

  
  GainSlider.setBounds(padding,  padding * 2, componentWidth,  componentHeight);
  generateButton.setBounds(padding, GainSlider.getBottom() + padding, componentWidth, componentHeight - 20);
  waveformComponent.setBounds(GainSlider.getRight() + padding, padding, componentWidth * 4, componentHeight * 2 + padding);
  midiKeyboardComponent.setBounds(padding, generateButton.getBottom() + padding, area.getWidth() - (2 * padding), componentHeight);

  GainLabel.setBounds(GainSlider.getX(), GainSlider.getY() - 20, GainSlider.getWidth(), GainSlider.getTextBoxHeight());
}



