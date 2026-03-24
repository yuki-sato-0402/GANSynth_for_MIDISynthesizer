# GANSynth_for_MIDISynthesizer
This project performs inference of the Google Magenta [GANSynth](https://magenta.withgoogle.com/gansynth) model within a JUCE-based audio environment, using [ONNX Runtime](https://github.com/anira-project/anira/tree/main).

Since the model takes a MIDI note number and a 256-dimensional latent noise vector as input, it can be adapted into a MIDI synthesizer. Based on this idea, this project explores using GANSynth as a playable instrument within a JUCE-based audio plugin.

The goal is to bridge machine learning-based sound generation with traditional MIDI workflows, enabling expressive and controllable synthesis driven by latent space manipulation.

## Features
- VST3 / AU / Standalone application support
- Playable as a MIDI synthesizer in any DAW that supports VST3 or AU
- The TensorFlow model is converted to ONNX for inference. 
- Since the model accepts a MIDI note number as input, the system generates samples (inference) every 6 semitones in a loop and fill in the gaps using pitch shifting.

## Demonstration
[Youtube<img width="668" height="394" alt="Screenshot 2026-03-18 at 22 37 58" src="https://github.com/user-attachments/assets/46580627-8ba0-4043-9eba-1a44296c30ff" />](https://youtu.be/S5SueJgJrvs) 


## 🛠️ Build Instructions
```
cd GANSynth_for_MIDISynthesizer
git submodule update --init --recursive
cd build
cmake ..
cmake --build .
```

## About conversion to ONNX models
To convert the original TensorFlow GANSynth model to ONNX, the following pipeline is required:

TensorFlow model → Freeze Graph → ONNX conversion

Although ONNX formally defines support for complex numbers, in practice this is limited. The [tf2onnx](https://github.com/onnx/tensorflow-onnx) converter (used to convert TensorFlow models to ONNX) currently supports up to opset 18, where complex number operations are not fully supported.　

As a result, parts of the model that rely on complex-valued operations—such as inverse FFT (iFFT)—cannot be directly converted and must be implemented separately on the host side.
In other words, the model's output itself only includes data in the spectral domain.

Additionally, the mel-to-linear transformation required for inverse FFT relies on a precomputed conversion matrix (computeMelToLinear). This matrix is ​​generated in Python and exported as a CSV file, which is then loaded and used within the JUCE implementation.

Several Python scripts are provided below.
- [gansynthFreezeGraph](pythonScripts/gansynthFreezeGraph.py) : A script to convert TF pretrained checkpoints to a Freeze Graph.

- [gansynth_onnxInference](pythonScripts/gansynth_onnxInference.py) : A script for inferring ONNX models in Python.

- [modelConfig.py](pythonScripts/modelConfig.py) : A script to check the shapes of the input and output tensors to the model.

## Acknowledgements
This project utilizes parts of the build system and inference logic inspired by the following [anira](https://github.com/anira-project/anira) projects:

* [anira/cmake/SetupOnnxRuntime.cmake](https://github.com/anira-project/anira/blob/main/cmake/SetupOnnxRuntime.cmake) - Used for automated ONNX Runtime installation and environment setup.
* [minimal-onnxruntime.cpp](https://github.com/anira-project/anira/blob/main/examples/minimal-inference/onnxruntime/minimal-onnxruntime.cpp) - Referenced for the project structure and C++ inference implementation patterns.

Special thanks to the authors for streamlining the complex process of integrating Neural Networks into JUCE applications.

## Other References
- [specgrams_helper_test.py](https://github.com/magenta/magenta/blob/main/magenta/models/gansynth/lib/specgrams_helper_test.py) : Used this as a reference when converting from the spectral domain to the time domain.

- [Tutorial: Build a MIDI synthesiser](https://juce.com/tutorials/tutorial_synth_using_midi_input/)

- [Let's build a synthesizer plug-in with C++ and the JUCE Framework!](https://youtube.com/playlist?list=PLLgJJsrdwhPwJimt5vtHtNmu63OucmPck&si=vfKCEvMZtt56co4B)
