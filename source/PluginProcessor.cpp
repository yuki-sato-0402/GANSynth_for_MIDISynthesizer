#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <complex>
#include <map>
#include <algorithm>

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
    std::cout << "Starting audio generation..." << std::endl;
    if (m_isGenerating) return;

    juce::Thread::launch([this]() {
        m_isGenerating = true;

        m_generatedNotes.clear();

        std::vector<float> latent(256);
        juce::Random r;
        for (auto& val : latent)
            val = nextGaussian(r);
        // Generate every 6 semitones from 21 to 108
        for (int note = 24; note <= 84; note += 6)
        {
            juce::AudioSampleBuffer tempBuffer;
            m_inference.generate(note, latent, tempBuffer);

            // Cache the generated audio for this note
            m_generatedNotes[note].makeCopyOf(tempBuffer);
        }

        m_isGenerating = false;

        // Notify listeners that generation is complete
        sendActionMessage("GenerationFinished");

        std::cout << "Audio generation completed." << std::endl;
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
            int midiNote = msg.getNoteNumber();
            m_lastMidiNote = midiNote;
              
            if (!m_generatedNotes.empty())
            {
            // lower_bound returns a pointer (iterator) pointing to the first element with a value greater than or equal to the currently played note 
                auto it = m_generatedNotes.lower_bound(midiNote);
                int closestNote = 60;
                
            // If the sound is higher than the overall level, use the highest-pitched cache
                if (it == m_generatedNotes.end()) {
                    closestNote = m_generatedNotes.rbegin()->first;
             // If the sound is lower than the overall level, use the lowest cache
                } else if (it == m_generatedNotes.begin()) {
                    closestNote = it->first;
                } else {
                    auto prev = std::prev(it);
                    //Choose the "closest sound" based on distance
                    if (midiNote - prev->first < it->first - midiNote)
                        closestNote = prev->first;
                    else
                        closestNote = it->first;
                }

                // Find a free voice
                Voice* voiceToUse = nullptr;
                for (auto& v : m_voices)
                {
                    if (!v.active)
                    {
                        voiceToUse = &v;
                        break;
                    }
                }
                
                // If no free voice, steal the one that is furthest along in playback
                if (voiceToUse == nullptr)
                {
                    voiceToUse = &m_voices[0];
                    for (auto& v : m_voices)
                        if (v.playIndex > voiceToUse->playIndex)
                            voiceToUse = &v;
                }

                voiceToUse->reset();
                voiceToUse->noteNumber = midiNote;
                voiceToUse->baseNote = closestNote;
                //Calculate the pitch of the complementary keys.
                voiceToUse->pitchRatio = std::pow(2.0, (double)(midiNote - closestNote) / 12.0);
                voiceToUse->active = true;
            }
        }
        else if (msg.isNoteOff())
        {
            int midiNote = msg.getNoteNumber();
            for (auto& v : m_voices)
            {
                if (v.active && v.noteNumber == midiNote)
                {
                    v.active = false; // Simple gate off (no release envelope in this version)
                }
            }
        }
    }

    buffer.clear();
    
    int numSamples = buffer.getNumSamples();
    const int fadeOutSamples = static_cast<int>(getSampleRate() * 0.05);

    // Manage which notes each voice is playing and up to which position
    for (auto& v : m_voices)
    {
        if (v.active && !m_generatedNotes.empty() && v.baseNote != -1)
        {
            auto& srcBuffer = m_generatedNotes[v.baseNote];
            int generatedSamples = srcBuffer.getNumSamples();

            if (v.playIndex < generatedSamples)
            {
                auto* src = srcBuffer.getReadPointer(0);
                
                // Use a temporary buffer for each voice to perform interpolation and then mix
                juce::AudioBuffer<float> tempVoiceBuffer(1, numSamples);
                tempVoiceBuffer.clear();
                auto* dest = tempVoiceBuffer.getWritePointer(0);
                
                int used = v.interpolator.process(v.pitchRatio, src + v.playIndex, dest, numSamples, generatedSamples - v.playIndex, 0);
                
                // Apply fade out at source buffer end
                int fadeStartPos = generatedSamples - fadeOutSamples;
                if (v.playIndex + used >= fadeStartPos)
                {
                    for (int i = 0; i < numSamples; ++i)
                    {
                        double currentInputPos = v.playIndex + (i * v.pitchRatio);
                        if (currentInputPos >= fadeStartPos)
                        {
                            float fade_gain = 1.0f - (float)(currentInputPos - fadeStartPos) / fadeOutSamples;
                            dest[i] *= std::max(0.0f, fade_gain);
                        }
                    }
                }

                // Mix into main buffer
                buffer.addFrom(0, 0, tempVoiceBuffer, 0, 0, numSamples);
                v.playIndex += used;
                
                if (used == 0 && v.playIndex >= generatedSamples - 5)
                    v.active = false;
            }
            else
            {
                v.active = false;
            }
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
