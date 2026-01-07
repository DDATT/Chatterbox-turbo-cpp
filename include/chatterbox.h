#include <iostream>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

class ChatterBox{
public:
    ChatterBox() = delete;
    ChatterBox(const std::string modelDir, bool useCuda);
    virtual ~ChatterBox();
    std::vector<int64_t> TokenizeText(std::string text);
    std::vector<int64_t> SynthesizeSpeechTokens(std::vector<int64_t> inputIds);
    std::vector<int16_t> synthesizeSpeech(std::vector<int64_t> generatedTokens);
    void LoadStyle(std::string styleDir);
    std::vector<float> LoadBinaryFile(std::string fileName);
    std::vector<int64_t> LoadBinaryFileInt64(std::string fileName);
    float repetitionPenalty = 1.2f; 
    const int64_t START_SPEECH_TOKEN = 6561;
    const int64_t STOP_SPEECH_TOKEN = 6562;
    const float MAX_WAV_VALUE = 32767.0f;
private:
    
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    Ort::Session conditionalDecoder;
    Ort::Session embedTokens;
    Ort::Session languageModel;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<float>condEmb;
    std::vector<int64_t>promptToken;
    std::vector<float>speakerEmbeddings;
    std::vector<float>speakerFeatures;

    std::array<const char*, 1> embedTokensInputNames = {"input_ids"};
    std::array<const char *, 1> bertEncoderOutputNames = {"inputs_embeds"};

    std::array<const char*, 51> languageModelInputNames = {
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value",
        "past_key_values.2.key",
        "past_key_values.2.value",
        "past_key_values.3.key",
        "past_key_values.3.value",
        "past_key_values.4.key",
        "past_key_values.4.value",
        "past_key_values.5.key",
        "past_key_values.5.value",
        "past_key_values.6.key",
        "past_key_values.6.value",
        "past_key_values.7.key",
        "past_key_values.7.value",
        "past_key_values.8.key",
        "past_key_values.8.value",
        "past_key_values.9.key",
        "past_key_values.9.value",
        "past_key_values.10.key",
        "past_key_values.10.value",
        "past_key_values.11.key",
        "past_key_values.11.value",
        "past_key_values.12.key",
        "past_key_values.12.value",
        "past_key_values.13.key",
        "past_key_values.13.value",
        "past_key_values.14.key",
        "past_key_values.14.value",
        "past_key_values.15.key",
        "past_key_values.15.value",
        "past_key_values.16.key",
        "past_key_values.16.value",
        "past_key_values.17.key",
        "past_key_values.17.value",
        "past_key_values.18.key",
        "past_key_values.18.value",
        "past_key_values.19.key",
        "past_key_values.19.value",
        "past_key_values.20.key",
        "past_key_values.20.value",
        "past_key_values.21.key",
        "past_key_values.21.value",
        "past_key_values.22.key",
        "past_key_values.22.value",
        "past_key_values.23.key",
        "past_key_values.23.value"
    };

    std::array<const char*, 49> languageModelOutputNames = {
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
        "present.2.key",
        "present.2.value",
        "present.3.key",
        "present.3.value",
        "present.4.key",
        "present.4.value",
        "present.5.key",
        "present.5.value",  
        "present.6.key",
        "present.6.value",      
        "present.7.key",
        "present.7.value",
        "present.8.key",
        "present.8.value",
        "present.9.key",
        "present.9.value",
        "present.10.key",
        "present.10.value",
        "present.11.key",
        "present.11.value",
        "present.12.key",
        "present.12.value",
        "present.13.key",
        "present.13.value",
        "present.14.key",
        "present.14.value",  
        "present.15.key",
        "present.15.value",   
        "present.16.key",
        "present.16.value",
        "present.17.key",
        "present.17.value",
        "present.18.key",
        "present.18.value",
        "present.19.key",
        "present.19.value",
        "present.20.key",
        "present.20.value",
        "present.21.key",
        "present.21.value",
        "present.22.key",
        "present.22.value",  
        "present.23.key",
        "present.23.value",   
    };

    std::array<const char*, 3> conditionalDecoderInputNames = {"speech_tokens", "speaker_embeddings", "speaker_features"};
    std::array<const char*, 1> conditionalDecoderOutputNames = {"waveform"};

    void applyRepetitionPenalty(float* logits, int64_t vocabSize, const std::vector<int64_t>& generatedTokens, float penalty);
};