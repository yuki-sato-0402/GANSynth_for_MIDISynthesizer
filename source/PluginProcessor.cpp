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
        ),
        pp_processor(inference_config), 
        custom_backend (inference_config),  
        inference_handler(pp_processor, inference_config, custom_backend)
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
    mutedSamples = sampleRate * 2;
    totalSamplesProcessed = 0;
    juce::dsp::ProcessSpec spec {sampleRate,
                                 static_cast<juce::uint32>(samplesPerBlock),
                                 static_cast<juce::uint32>(getTotalNumInputChannels())};

    anira::HostConfig host_config {
        static_cast<float>(samplesPerBlock),
        static_cast<float>(sampleRate),
    };

    //inference_handler.set_non_realtime(true);
    std::cout << "$Preparing GANSynth for MIDI Synthesizer...$" << std::endl;
    inference_handler.prepare(host_config);
    inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
    inference_handler.set_non_realtime(true);
    
    std::cout << "$Inference latency (in samples): " << inference_handler.get_latency() << "$" << std::endl;
    int new_latency = (int) inference_handler.get_latency();

    gain.prepare(spec);
    gain.setGainLinear(gainParam);

    setLatencySamples(new_latency);

    midiMessageCollector.reset(sampleRate);

    std::cout << "$Using Custom GANSynth backend for inference.$" << std::endl;
    //inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);
    // In prepareToPlay(), after prepare():
    std::cout << "[DEBUG] Active inferences: " << inference_handler.get_available_samples(0) << std::endl;
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

void GANSynth_for_MIDISynthesizer_Processor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    midiMessageCollector.removeNextBlockOfMessages(midiMessages, buffer.getNumSamples());
    
    for (const auto metadata : midiMessages)
    {
        const auto msg = metadata.getMessage();

        if (msg.isNoteOn())
        {
            m_lastMidiNote = msg.getNoteNumber();
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

        if (playIdx < generatedSamples)
        {
            int samplesToCopy = std::min(numSamples, generatedSamples - playIdx);
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            {
                buffer.copyFrom(channel, 0, m_generatedAudio, 0, playIdx, samplesToCopy);
            }
            m_playIndex += samplesToCopy;
        }
        else
        {
            m_isPlaying = false;
        }
    }

    juce::dsp::AudioBlock<float> audioBlock(buffer);
    gain.process(juce::dsp::ProcessContextReplacing<float>(audioBlock));

    totalSamplesProcessed += buffer.getNumSamples();
}

void GANSynth_for_MIDISynthesizer_Processor::triggerInference(int midiNote)
{
    if (m_isGenerating) return;

    m_isGenerating = true;

    juce::Thread::launch([this, midiNote]()
    {
        std::cout << "Starting GANSynth inference for MIDI note: " << midiNote << std::endl;
        
        // 1. Prepare Inputs
         
        // Tensor 0: Note [1]
        // Note: matches GANSynthModelConfig.h m_tensor_input_size[0]
        std::vector<float> input0Data(1, static_cast<float>(midiNote));
        const float* input0Ptr = input0Data.data();
        const float* input0ChannelPtrs[] = { input0Ptr };

        // Tensor 1: Latent Noise [256]
        // Note: matches GANSynthModelConfig.h m_tensor_input_size[1]
        std::vector<float> input1Data(256);
        juce::Random rand;
        for (int i = 0; i < 256; ++i) {
            input1Data[i] = rand.nextFloat() * 2.0f - 1.0f;
        }
        const float* input1Ptr = input1Data.data();
        const float* input1ChannelPtrs[] = { input1Ptr };

        // Bundle inputs (each tensor -> array of channel pointers)
        const float* const* allInputPtrs[] = { input0ChannelPtrs, input1ChannelPtrs };
        size_t allInputSizes[] = { (size_t)input0Data.size(), (size_t)input1Data.size() };
        std::cout << "(size_t)input0Data.size(): " << (size_t)input0Data.size() << ", (size_t)input1Data.size(): " << (size_t)input1Data.size() << std::endl;
        
         // Push the input data to the custom backend processor
        //inference_handler.push_data(allInputPtrs, allInputSizes);  
     
        // Prepare Output Buffer
        const int triggerSize = 128 * 1024 * 2; // matches GANSynthModelConfig.h m_tensor_output_size[0]
        std::vector<float> outputRaw(triggerSize, 0.0f);
        float* output0Ptr = outputRaw.data();
        float* output0ChannelPtrs[] = { output0Ptr };
        float* const* allOutputPtrs[] = { output0ChannelPtrs };
        size_t allOutputSizes[] = { (size_t)triggerSize };

        // Populate the output buffer with inference results
        //auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10); 
        //size_t popped = inference_handler.pop_data(output0ChannelPtrs, allOutputSizes[0], timeout, 0);    


        //std::cout << "$[PluginProcessor] Input size: " << allInputSizes[0] << ", " << allInputSizes[1] << "$" << std::endl;
        // 3. Inference
        //size_t tensor_index is an index used to specify a particular tensor among multiple tensors (inputs and outputs). In this case, since we have only one output tensor, we can set it to 0.
        //size_t tensor_index = 0;
        inference_handler.process(allInputPtrs, allInputSizes, allOutputPtrs, allOutputSizes);
       
        // 4. iSTFT Post-processing
        const int numFrames = 128;
        const int numMelBins = 1024;
        const int fftSize = 2048;
        const int hopSize = 512;
        const int numMagBins = fftSize / 2 + 1;
        const int totalSamples = numFrames * hopSize + fftSize;

        m_generatedAudio.setSize(1, totalSamples);
        m_generatedAudio.clear();
        float* audioPtr = m_generatedAudio.getWritePointer(0);

        juce::dsp::FFT fft(11);
        std::vector<float> window(fftSize);
        for (int i = 0; i < fftSize; ++i) window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (fftSize - 1)));
        
        std::vector<std::complex<float>> freqData(fftSize);
        std::vector<float> phase(numMelBins, 0.0f);

        for (int f = 0; f < numFrames; ++f) {
            const float* frameData = outputRaw.data() + (f * numMelBins * 2);
            std::fill(freqData.begin(), freqData.end(), std::complex<float>(0, 0));
            
            for (int m = 0; m < numMelBins; ++m) {
                float logMelMag = frameData[m * 2];
                float melIFreq = frameData[m * 2 + 1];
                phase[m] += melIFreq * M_PI;
                float mag = std::exp(logMelMag);
                int linearBin = (int)((float)m / numMelBins * numMagBins);
                if (linearBin < numMagBins) freqData[linearBin] += std::polar(mag, phase[m]);
            }

            fft.perform(freqData.data(), (juce::dsp::Complex<float>*)freqData.data(), true);
            
            int offset = f * hopSize;
            for (int i = 0; i < fftSize; ++i) {
                if (offset + i < totalSamples) audioPtr[offset + i] += freqData[i].real() * window[i];
            }
        }

        // Normalization
        float maxMag = m_generatedAudio.getMagnitude(0, 0, totalSamples);
        m_generatedAudio.applyGain(1.0f / (maxMag + 1e-8f));

        std::cout << "Inference and Reconstruction completed." << std::endl;
        m_isGenerating = false;
    });
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
