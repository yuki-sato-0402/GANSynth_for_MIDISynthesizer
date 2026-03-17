#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <complex>

//==============================================================================
GANSynth_for_MIDISynthesizer_Processor::GANSynth_for_MIDISynthesizer_Processor() 
        : AudioProcessor (BusesProperties()
                       .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                       .withOutput ("Output", juce::AudioChannelSet::mono(), true)
                       ),
        apvts(*this, nullptr, juce::Identifier("PARAMETERS"),
        juce::AudioProcessorValueTreeState::ParameterLayout {
        std::make_unique<juce::AudioParameterFloat>(juce::ParameterID { "outputGain",  1}, "OutputGain",
        juce::NormalisableRange<float>(0.f, 1.f, 0.01f), 0.5f), 
        }
        )
{
    apvts.addParameterListener("outputGain", this);
    gainParam = *apvts.getRawParameterValue("outputGain");
}

//==============================================================================
const juce::String GANSynth_for_MIDISynthesizer_Processor::getName() const
{
    return JucePlugin_Name;
}

bool GANSynth_for_MIDISynthesizer_Processor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool GANSynth_for_MIDISynthesizer_Processor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool GANSynth_for_MIDISynthesizer_Processor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double GANSynth_for_MIDISynthesizer_Processor::getTailLengthSeconds() const
{
    return 0.0;
}

int GANSynth_for_MIDISynthesizer_Processor::getNumPrograms()
{
    return 1;
}

int GANSynth_for_MIDISynthesizer_Processor::getCurrentProgram()
{
    return 0;
}

void GANSynth_for_MIDISynthesizer_Processor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String GANSynth_for_MIDISynthesizer_Processor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void GANSynth_for_MIDISynthesizer_Processor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void GANSynth_for_MIDISynthesizer_Processor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec {sampleRate,
                                 static_cast<juce::uint32>(samplesPerBlock),
                                 static_cast<juce::uint32>(getTotalNumInputChannels())};

    gain.prepare(spec);
    gain.setGainLinear(gainParam);

    
    midiMessageCollector.reset(sampleRate);
    
    m_inference.setTargetSampleRate(sampleRate);
    m_inference.prepare(GANSynth_MODEL_DIR "gansynth.onnx");
    m_inference.loadMel2lFromCsv(GANSynth_MODEL_DIR "mel2l_matrix.csv");
}

float GANSynth_for_MIDISynthesizer_Processor::nextGaussian(juce::Random& r) {
    // Box-Muller transform
    float u1 = r.nextFloat();
    float u2 = r.nextFloat();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * juce::MathConstants<float>::pi * u2);
}

void GANSynth_for_MIDISynthesizer_Processor::generateAudio()
{
    if (m_isGenerating) return;

    juce::Thread::launch([this]() {
        m_isGenerating = true;
        
        std::vector<float> latent(256);
        juce::Random r;
        for (auto& val : latent)
            val = nextGaussian(r);
        
        m_inference.generate(m_lastMidiNote, latent, m_generatedAudio);

        std::cout << "Generated audio with " << m_generatedAudio.getNumSamples() << " samples." << std::endl;
        
        m_isGenerating = false;
        
        // Notify listeners that generation is complete
        sendActionMessage("GenerationFinished");
    });
}

void GANSynth_for_MIDISynthesizer_Processor::releaseResources()
{
}

bool GANSynth_for_MIDISynthesizer_Processor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
        return false;

    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;
    else
        return true;
}

void GANSynth_for_MIDISynthesizer_Processor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    midiMessageCollector.removeNextBlockOfMessages(midiMessages, buffer.getNumSamples());
    
    for (const auto metadata : midiMessages)
    {
        const auto msg = metadata.getMessage();

        if (msg.isNoteOn())
        {
            m_playIndex = 0;
            m_isPlaying = true;
        }
    }

    buffer.clear();
    if (m_isPlaying && m_generatedAudio.getNumSamples() > 0)
    {
        int numSamples = buffer.getNumSamples();
        int generatedSamples = m_generatedAudio.getNumSamples();
        int playIdx = m_playIndex.load();
        const int fadeOutSamples = static_cast<int>(getSampleRate() * 0.02);
        int fadeStartPos = generatedSamples - fadeOutSamples;

        if (playIdx < generatedSamples){
            int samplesToCopy = std::min(numSamples, generatedSamples - playIdx);

            for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            {
                auto* dest = buffer.getWritePointer(channel);
                auto* src  = m_generatedAudio.getReadPointer(0);

                for (int i = 0; i < samplesToCopy; ++i)
                {
                    int globalSample = playIdx + i;

                    float fade_gain = 1.0f;

                    if (globalSample >= fadeStartPos)
                    {
                        float fadePos = (float)(globalSample - fadeStartPos) / fadeOutSamples;
                        fade_gain = 1.0f - fadePos;
                    }

                    dest[i] = src[globalSample] * fade_gain;
                }
            }

            m_playIndex += samplesToCopy;
        }else
        {
            m_isPlaying = false;
        }
    }

    juce::dsp::AudioBlock<float> audioBlock(buffer);
    gain.process(juce::dsp::ProcessContextReplacing<float>(audioBlock));
}


//==============================================================================
bool GANSynth_for_MIDISynthesizer_Processor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* GANSynth_for_MIDISynthesizer_Processor::createEditor()
{
    return new GANSynth_for_MIDISynthesizer_ProcessorEditor (*this,  apvts);
}

//==============================================================================
void GANSynth_for_MIDISynthesizer_Processor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void GANSynth_for_MIDISynthesizer_Processor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

void GANSynth_for_MIDISynthesizer_Processor::parameterChanged(const juce::String &parameterID, float newValue) {
    if(parameterID == "outputGain") 
    {
        gainParam = newValue;
        gain.setGainLinear(gainParam);
        std::cout << "OutputGain changed to: " << newValue << std::endl;
    }
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new GANSynth_for_MIDISynthesizer_Processor();
}
