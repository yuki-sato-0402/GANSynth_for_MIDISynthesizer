#include "GANSynthInference.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>
#include <algorithm>
#include <numeric>

GANSynthInference::GANSynthInference()
    : m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{
    m_sessionOptions.SetIntraOpNumThreads(1);
    m_fft = std::make_unique<juce::dsp::FFT>(11); // 2^11 = 2048
    //computeMelToLinearMatrix();
}

GANSynthInference::~GANSynthInference()
{
}

void GANSynthInference::prepare(const juce::String& modelPath)
{
    if (m_session) return; // Already loaded

#ifdef _WIN32
    std::wstring modelWideStr = modelPath.toWideCharPointer();
    m_session = std::make_unique<Ort::Session>(m_env, modelWideStr.c_str(), m_sessionOptions);
#else
    m_session = std::make_unique<Ort::Session>(m_env, modelPath.toRawUTF8(), m_sessionOptions);
#endif
}

bool GANSynthInference::loadMel2lFromCsv(const juce::String& csvPath)
{
    std::ifstream file(csvPath.toStdString());
    if (!file.is_open()) {
        std::cerr << "[GANSynth] Cannot open mel2l CSV: " << csvPath << std::endl;
        return false;
    }
 
    std::vector<float> data;
    data.reserve(static_cast<size_t>(m_nMel * m_nMag));
 
    std::string line;
    int rowCount = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        int colCount = 0;
        while (std::getline(ss, cell, ',')) {
            try {
                data.push_back(std::stof(cell));
            } catch (...) {
                std::cerr << "[GANSynth] Parse error at row=" << rowCount
                          << " col=" << colCount << ": '" << cell << "'" << std::endl;
                return false;
            }
            ++colCount;
        }
        ++rowCount;
    }
 
    // Check if the loaded data matches the expected dimensions
    const int expected = m_nMel * m_nMag; // 1024 * 1025 = 1,049,600
    if ((int)data.size() != expected) {
        std::cerr << "[GANSynth] mel2l CSV size mismatch: got " << data.size()
                  << " values, expected " << expected
                  << " (" << m_nMel << " rows x " << m_nMag << " cols)" << std::endl;
        return false;
    }
 
    m_mel2l = std::move(data);
    m_mel2lLoaded = true;
 
    std::cout << "[GANSynth] mel2l loaded from CSV: " << csvPath
              << "  shape=[" << rowCount << ", " << m_nMag << "]" << std::endl;
 
    // Debug print the top-left 5x5 block of the matrix
    std::cout << "[GANSynth] mel2l[:5,:5]:" << std::endl;
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i)
            std::cout << std::fixed << std::setprecision(3)<< m_mel2l[static_cast<size_t>(j * m_nMag + i)] << " ";
        std::cout << std::endl;
    }
 
    return true;
}


void GANSynthInference::generate(int midiNote, const std::vector<float>& latentVector, juce::AudioBuffer<float>& outputBuffer)
{
    std::cout << "Generating audio for MIDI note: " << midiNote << std::endl;
    if (!m_session) return;

    int label = midiNote - 24;
    int32_t labelData = (int32_t)label;

    std::vector<int64_t> labelShape  = {1};
    std::vector<int64_t> latentShape = {1, 256};

    Ort::Value labelTensor = Ort::Value::CreateTensor<int32_t>(
        m_memoryInfo, &labelData, 1, labelShape.data(), labelShape.size());

    std::vector<float> latentData = latentVector;
    if (latentData.size() != 256) latentData.resize(256, 0.0f);

    Ort::Value latentTensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo, latentData.data(), latentData.size(), latentShape.data(), latentShape.size());

    const char* inputNames[]  = {"Placeholder:0", "Placeholder_1:0"};
    const char* outputNames[] = {"Generator_1/truediv:0"};
    Ort::Value inputs[] = {std::move(labelTensor), std::move(latentTensor)};

    try {
        auto outputs = m_session->Run(Ort::RunOptions{nullptr}, inputNames, inputs, 2, outputNames, 1);

        auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output shape: "
                  << outputShape[0] << ", " << outputShape[1] << ", "
                  << outputShape[2] << ", " << outputShape[3] << std::endl;

        float* outputData = outputs[0].GetTensorMutableData<float>();
        postProcess(outputData, outputBuffer);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
    }
}

//static float hertz_to_mel(float hz)  { return 1127.0f * std::log(1.0f + hz / 700.0f); }
//static float mel_to_hertz(float mel) { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }

// ─────────────────────────────────────────────────────────────────────────────
// computeMelToLinearMatrix
//
//   Python reference (specgrams_helper.py / spectral_ops.py):
//     m     = linear_to_mel_weight_matrix(N_MEL, N_MAG, SR)  # [N_MAG, N_MEL]
//     m_t   = m.T                                             # [N_MEL, N_MAG]
//     p     = m @ m_t                                         # [N_MAG, N_MAG]
//     d     = 1 / sum(p, axis=0)           # column-wise sum → [N_MAG]
//     mel2l = m_t @ diag(d)                                   # [N_MEL, N_MAG]
//
//   Storage layout:
//     M      : row-major [N_MAG × N_MEL],  M[i,j]     = M_flat[i*N_MEL + j]
//     m_mel2l: row-major [N_MEL × N_MAG],  mel2l[j,i] = m_mel2l[j*N_MAG + i]
// ─────────────────────────────────────────────────────────────────────────────
//void GANSynthInference::computeMelToLinearMatrix()
//{
//    const float nyquist       = m_sampleRate / 2.0f;
//    const int   bands_to_zero = 1;

//    std::vector<float> linear_frequencies(m_nMag);
//    for (int i = 0; i < m_nMag; ++i)
//        linear_frequencies[i] = (float)i * nyquist / (float)(m_nMag - 1);

//    std::vector<float> band_edges_mel(m_nMel + 2);
//    {
//        float mel_min = hertz_to_mel(0.0f);
//        float mel_max = hertz_to_mel(nyquist);
//        for (int i = 0; i < m_nMel + 2; ++i)
//            band_edges_mel[i] = mel_min + (float)i * (mel_max - mel_min) / (float)(m_nMel + 1);
//    }

//    std::vector<float> lower_edge_mel(m_nMel), center_mel(m_nMel), upper_edge_mel(m_nMel);
//    for (int i = 0; i < m_nMel; ++i) {
//        lower_edge_mel[i] = band_edges_mel[i];
//        center_mel[i]     = band_edges_mel[i + 1];
//        upper_edge_mel[i] = band_edges_mel[i + 2];
//    }

//    // Minimum bandwidth enforcement
//    {
//        float freq_res = nyquist / (float)(m_nMag - 1);
//        float freq_th  = 1.5f * freq_res;
//        std::cout << "Minimum frequency threshold for mel bands: " << freq_th << " Hz" << std::endl;
//        for (int i = 0; i < m_nMel; ++i) {
//            float lower_hz = mel_to_hertz(lower_edge_mel[i]);
//            float upper_hz = mel_to_hertz(upper_edge_mel[i]);
//            if (upper_hz - lower_hz < freq_th) {
//                float center_hz = mel_to_hertz(center_mel[i]);
//                float rhs = 0.5f * freq_th / (center_hz + 700.0f);
//                float dm  = 1127.0f * std::log(rhs + std::sqrt(1.0f + rhs * rhs));
//                lower_edge_mel[i] = center_mel[i] - dm;
//                upper_edge_mel[i] = center_mel[i] + dm;
//            }
//        }
//    }

//    // Build M [N_MAG × N_MEL]  (row 0 = DC bin stays zero)
//    std::vector<float> M(m_nMag * m_nMel, 0.0f);
//    for (int j = 0; j < m_nMel; ++j) {
//        float l_hz = mel_to_hertz(lower_edge_mel[j]);
//        float c_hz = mel_to_hertz(center_mel[j]);
//        float u_hz = mel_to_hertz(upper_edge_mel[j]);
//        for (int i = bands_to_zero; i < m_nMag; ++i) {
//            float f           = linear_frequencies[i];
//            float lower_slope = (f - l_hz) / (c_hz - l_hz);
//            float upper_slope = (u_hz - f) / (u_hz - c_hz);
//            M[i * m_nMel + j] = std::max(0.0f, std::min(lower_slope, upper_slope));
//        }
//    }

//    // d[i] = 1 / sum(P, axis=0)[i]
//    // sum(P, axis=0)[i] = sum_row P[row,i] = sum_k M[i,k] * (sum_row M[row,k])
//    std::vector<float> sum_M_cols(m_nMel, 0.0f);
//    for (int k = 0; k < m_nMel; ++k)
//        for (int row = 0; row < m_nMag; ++row)
//            sum_M_cols[k] += M[row * m_nMel + k];

//    std::vector<float> sum_P_col(m_nMag, 0.0f);
//    for (int i = 0; i < m_nMag; ++i)
//        for (int k = 0; k < m_nMel; ++k)
//            sum_P_col[i] += M[i * m_nMel + k] * sum_M_cols[k];

//    // mel2l[j, i] = M[i, j] * d[i]
//    m_mel2l.assign(m_nMel * m_nMag, 0.0f);
//    for (int j = 0; j < m_nMel; ++j)
//        for (int i = 0; i < m_nMag; ++i) {
//            float d = (std::abs(sum_P_col[i]) > 1e-8f) ? (1.0f / sum_P_col[i]) : 0.0f;
//            m_mel2l[j * m_nMag + i] = M[i * m_nMel + j] * d;
//        }
//}

// ─────────────────────────────────────────────────────────────────────────────
// postProcess
//
//  Correct pipeline (matches original specgrams_helper.py):
//
//    rawOutput [T, N_MEL, 2]:
//      ch0 = logmelmag2   ch1 = mel_IF
//
//  Magnitude:
//    mel_power[t,j] = exp(logmelmag2[t,j])
//    pow_lin[t,i]   = dot(mel_power[t,:], mel2l)     → [N_MAG]
//    mag_lin[t,i]   = sqrt(max(0, pow_lin))
//
//  Phase  ★ KEY: transform IF to linear domain FIRST, then cumsum ★
//    lin_IF[t,i]    = dot(mel_IF[t,:], mel2l)         → [N_MAG]
//    phase[t,i]     = cumsum_t( lin_IF * π )
//
//  ISTFT:
//    stft[t,i]      = polar(mag_lin, phase)
//    audio          = IFFT + OLA (Hann window, center=True)
// ─────────────────────────────────────────────────────────────────────────────
void GANSynthInference::postProcess(const float* rawOutput, juce::AudioBuffer<float>& outputBuffer)
{
    // librosa istft output length with center=True:
    //   (n_frames - 1) * hop_length + win_length
    const int total_samples = (m_numFrames - 1) * m_hopLength;

    juce::AudioSampleBuffer tempBuffer(1, total_samples);
    tempBuffer.clear();
    float* tmp_ptr = tempBuffer.getWritePointer(0);

    // Linear-domain phase accumulator per bin [N_MAG]
    // This is the cumsum of (linear_IF * π) across time frames
    std::vector<float> lin_phase_acc(static_cast<size_t>(m_nMag), 0.0f);

    std::vector<float> mel_power(static_cast<size_t>(m_nMel));
    std::vector<float> mel_IF(static_cast<size_t>(m_nMel));

    // juce::dsp::FFT::perform needs buffers of size 2*N_FFT
    std::vector<std::complex<float>> fft_in (static_cast<size_t>(m_nFFT * 2), {0.0f, 0.0f});
    std::vector<std::complex<float>> fft_out(static_cast<size_t>(m_nFFT * 2), {0.0f, 0.0f});

    std::vector<float> norm(static_cast<size_t>(total_samples), 0.0f);

    // Hann synthesis window
    std::vector<float> window(static_cast<size_t>(m_nFFT));
    juce::dsp::WindowingFunction<float>::fillWindowingTables(
        window.data(), (size_t)m_nFFT, juce::dsp::WindowingFunction<float>::hann);

    // librosa center=True: frame t centered at t * hop_length
    const int center_offset = m_nFFT / 2;

    for (int t = 0; t < m_numFrames; ++t)
    {
        // rawOutput layout: [T, N_MEL, 2] interleaved per mel bin
        const float* frame = rawOutput + (t * m_nMel * 2);

        // ── Step 1: Read mel-domain values ───────────────────────────────────
        for (int j = 0; j < m_nMel; ++j) {
            mel_power[static_cast<size_t>(j)] = std::exp(frame[static_cast<size_t>(j * 2 + 0)]);  // exp(logmelmag2) = mel power
            mel_IF[static_cast<size_t>(j)]    = frame[static_cast<size_t>(j * 2 + 1)];            // mel instantaneous frequency
        }

        // ── Step 2: Mel → Linear for power and IF ────────────────────────────
        std::fill(fft_in.begin(), fft_in.end(), std::complex<float>(0.0f, 0.0f));

        for (int i = 0; i < m_nMag; ++i) {
            float pow_lin = 0.0f;
            float if_lin  = 0.0f;

            for (int j = 0; j < m_nMel; ++j) {
                float w  = m_mel2l[static_cast<size_t>(j * m_nMag + i)];
                pow_lin += mel_power[static_cast<size_t>(j)] * w;
                if_lin  += mel_IF[static_cast<size_t>(j)]    * w;   // ← IF transformed to linear domain
            }

            // Power → Magnitude
            float mag_lin = std::sqrt(std::max(0.0f, pow_lin));

            // ★ Accumulate phase in LINEAR domain (cumsum of linear_IF * π)
            lin_phase_acc[static_cast<size_t>(i)] += if_lin * juce::MathConstants<float>::pi;

            fft_in[static_cast<size_t>(i)] = std::polar(mag_lin, lin_phase_acc[static_cast<size_t>(i)]);
        }

        // Hermitian symmetry for real-valued IFFT
        for (int i = 1; i < m_nFFT / 2; ++i)
            fft_in[static_cast<size_t>(m_nFFT - i)] = std::conj(fft_in[static_cast<size_t>(i)]);

        // Nyquist bin must be real
        fft_in[static_cast<size_t>(m_nFFT / 2)] = std::complex<float>(fft_in[static_cast<size_t>(m_nFFT / 2)].real(), 0.0f);

        // ── Step 3: IFFT (separate in/out buffers) ────────────────────────────
        m_fft->perform(fft_in.data(), fft_out.data(), true);

        // ── Step 4: Overlap-Add ───────────────────────────────────────────────
        int frame_start = t * m_hopLength - center_offset;
        for (int i = 0; i < m_nFFT; ++i) {
            int idx = frame_start + i;
            if (idx >= 0 && idx < total_samples) {
                tmp_ptr[static_cast<size_t>(idx)] += fft_out[static_cast<size_t>(i)].real() * window[static_cast<size_t>(i)];
                norm[static_cast<size_t>(idx)]    += window[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
            }
        }
    }

    // ── Step 5: OLA normalization ─────────────────────────────────────────────
    for (int i = 0; i < total_samples; ++i)
        if (norm[static_cast<size_t>(i)] > 1e-10f)
            tmp_ptr[static_cast<size_t>(i)] /= norm[static_cast<size_t>(i)];

    // ── Step 6: Resample if needed ────────────────────────────────────────────
    std::cout << "Source SR: " << m_sampleRate
              << "  Target SR: " << m_targetSampleRate << std::endl;

    if (std::abs(m_targetSampleRate - (double)m_sampleRate) > 0.1)
    {
        // LagrangeInterpolator::process(speedRatio, src, dst, numDstSamples)
        // speedRatio = srcSampleRate / dstSampleRate  (= 16000/44100 ≈ 0.363)
        double speedRatio    = (double)m_sampleRate / m_targetSampleRate;
        int    targetSamples = (int)std::round((double)total_samples / speedRatio);

        outputBuffer.setSize(1, targetSamples, false, true, false);
        outputBuffer.clear();
        float* out_ptr = outputBuffer.getWritePointer(0);

        resampler.reset();
        resampler.process(speedRatio, tmp_ptr, out_ptr, targetSamples);

        std::cout << "Resampled to " << targetSamples << " samples." << std::endl;
    }
    else
    {
        outputBuffer.setSize(1, total_samples, false, true, false);
        outputBuffer.clear();
        outputBuffer.copyFrom(0, 0, tempBuffer, 0, 0, total_samples);
    }

    std::cout << "Post-processing completed." << std::endl;
}