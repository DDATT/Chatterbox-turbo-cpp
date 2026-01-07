#include "chatterbox.h"

ChatterBox::ChatterBox(const std::string modelDir, bool useCuda)
    : env_(nullptr),
      sessionOptions_(),
      conditionalDecoder(nullptr),
      embedTokens(nullptr),
      languageModel(nullptr) {
    
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Chatterbox-turbo");
    env_.DisableTelemetryEvents();                       

    if (useCuda) {
        // Use CUDA provider
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
    }
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    sessionOptions_.DisableCpuMemArena();
    sessionOptions_.DisableMemPattern();
    sessionOptions_.DisableProfiling();

    std::string conditionalDecoderPathString = modelDir + "/conditional_decoder.onnx";
    std::string embedTokensPathString = modelDir + "/embed_tokens.onnx";
    std::string languageModelPathString = modelDir + "/language_model.onnx";
    #ifdef _WIN32
    std::wstring conditionalDecoderPath_wstr = std::wstring(conditionalDecoderPathString.begin(), conditionalDecoderPathString.end());
    std::wstring embedTokensPath_wstr = std::wstring(embedTokensPathString.begin(), embedTokensPathString.end());
    std::wstring languageModelPath_wstr = std::wstring(languageModelPathString.begin(), languageModelPathString.end());
    auto conditionalDecoderPath = conditionalDecoderPath_wstr.c_str();
    auto embedTokensPath = embedTokensPath_wstr.c_str();
    auto languageModelPath = languageModelPath_wstr.c_str();
    #else
    auto conditionalDecoderPath = conditionalDecoderPathString.c_str();
    auto embedTokensPath = embedTokensPathString.c_str();
    auto languageModelPath = languageModelPathString.c_str();
    #endif

    conditionalDecoder = Ort::Session(env_, conditionalDecoderPath, sessionOptions_);
    embedTokens = Ort::Session(env_, embedTokensPath, sessionOptions_);
    languageModel = Ort::Session(env_, languageModelPath, sessionOptions_);
}

ChatterBox::~ChatterBox() {}

void ChatterBox::LoadStyle(std::string styleDir) {
    std::string condEmbPath = styleDir + "/cond_emb.bin";
    condEmb = LoadBinaryFile(condEmbPath);

    std::string promptTokenPath = styleDir + "/prompt_token.bin";
    promptToken = LoadBinaryFileInt64(promptTokenPath);

    std::string speakerEmbeddingsPath = styleDir + "/speaker_embeddings.bin";
    speakerEmbeddings = LoadBinaryFile(speakerEmbeddingsPath);

    std::string speakerFeaturesPath = styleDir + "/speaker_features.bin";
    speakerFeatures = LoadBinaryFile(speakerFeaturesPath);
}

std::vector<float> ChatterBox::LoadBinaryFile(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "File not found!" << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));

    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }

    return {};
}


std::vector<int64_t> ChatterBox::LoadBinaryFileInt64(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "File not found!" << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<int64_t> buffer(size / sizeof(int64_t));

    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }

    return {};
}

std::vector<int64_t> ChatterBox::SynthesizeSpeechTokens(std::vector<int64_t> inputIds) {
    std::vector<int64_t> generatedTokens;
    generatedTokens.push_back(START_SPEECH_TOKEN);

    std::vector<int64_t> embedTokensInputsDim{1, static_cast<int64_t>(inputIds.size())};
    std::vector<Ort::Value> embedTokensInputTensors;
    embedTokensInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, inputIds.data(), inputIds.size(),
            embedTokensInputsDim.data(), embedTokensInputsDim.size()));
    int64_t condEmbLength = int64_t(condEmb.size() / 1024);
    int64_t currentSeqLen = static_cast<int64_t>(inputIds.size()) + condEmbLength;
    int64_t currentPosition = currentSeqLen - 1; 

    std::vector<Ort::Value> languageModelInputTensors;
    std::vector<Ort::Value> pastKeyValues; 
    
    int64_t nextTokenId = 0;
    for (int i = 0; i < 1024; i++) {
        std::vector<float> currentEmbedsData;
        std::vector<int64_t> currentEmbedsShape;

        if (i == 0) {
            // Get embedding from input text
            std::vector<int64_t> embedTokensInputsDim{1, static_cast<int64_t>(inputIds.size())};
            Ort::Value embedTokensInput = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, inputIds.data(), inputIds.size(),
                embedTokensInputsDim.data(), embedTokensInputsDim.size());
            
            auto inputsEmbedsOutput = embedTokens.Run(Ort::RunOptions{nullptr}, 
                embedTokensInputNames.data(), &embedTokensInput, 1, 
                bertEncoderOutputNames.data(), bertEncoderOutputNames.size());

            const float* promptEmbedsData = inputsEmbedsOutput.front().GetTensorData<float>();
            size_t promptEmbedsSize = inputsEmbedsOutput.front().GetTensorTypeAndShapeInfo().GetElementCount();

            currentEmbedsData = condEmb;
            currentEmbedsData.insert(currentEmbedsData.end(), promptEmbedsData, promptEmbedsData + promptEmbedsSize);
            
            currentEmbedsShape = {1, static_cast<int64_t>(inputIds.size()) + condEmbLength, 1024};

        } 
        else {
            // Get embedding for the next generated token            
            std::vector<int64_t> nextInputIdVec = {nextTokenId};
            std::vector<int64_t> embedTokensInputsDim{1, 1}; // Batch 1, Seq 1
            
            Ort::Value embedTokensInput = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, nextInputIdVec.data(), nextInputIdVec.size(),
                embedTokensInputsDim.data(), embedTokensInputsDim.size());

            auto inputsEmbedsOutput = embedTokens.Run(Ort::RunOptions{nullptr}, 
                embedTokensInputNames.data(), &embedTokensInput, 1, 
                bertEncoderOutputNames.data(), bertEncoderOutputNames.size());
            
            const float* newEmbedData = inputsEmbedsOutput.front().GetTensorData<float>();
            size_t newEmbedSize = inputsEmbedsOutput.front().GetTensorTypeAndShapeInfo().GetElementCount();

            currentEmbedsData.assign(newEmbedData, newEmbedData + newEmbedSize);
            currentEmbedsShape = {1, 1, 1024}; // Each iteration just generated one token
        }

        // prepare inputs for language model
        languageModelInputTensors.clear();

        // Input 0: inputs_embeds
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, currentEmbedsData.data(), currentEmbedsData.size(),
            currentEmbedsShape.data(), currentEmbedsShape.size()));

        // Input 1: attention_mask
        std::vector<int64_t> currentMask(currentSeqLen, 1);
        std::vector<int64_t> currentMaskShape{1, currentSeqLen};
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, currentMask.data(), currentMask.size(),
            currentMaskShape.data(), currentMaskShape.size()));

        // Input 2: position_ids
        std::vector<int64_t> currentPosIds;
        std::vector<int64_t> currentPosIdsShape;
        
        if (i == 0) {
            // i=0: 0, 1, 2, ..., seqLen-1
            currentPosIds.resize(currentSeqLen);
            for(int k=0; k<currentSeqLen; ++k) currentPosIds[k] = k;
            currentPosIdsShape = {1, currentSeqLen};
        } else {
            currentPosition++;
            currentPosIds.push_back(currentPosition);
            currentPosIdsShape = {1, 1};
        }
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, currentPosIds.data(), currentPosIds.size(),
            currentPosIdsShape.data(), currentPosIdsShape.size()));

        // Input 3..50: past_key_values
        if (i == 0) {
            // At first iteration, past KV is setted to zeros
            for (int j = 0; j < 48; j++) {
                std::vector<int64_t> pastShape = {1, 16, 0, 64};
                std::vector<float> pastData(0.0); 
                pastKeyValues.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, pastData.data(), pastData.size(),
                    pastShape.data(), pastShape.size()));
            }
        }
        
        // Push KV Cache into input list
        for (auto& kv : pastKeyValues) {
            languageModelInputTensors.push_back(std::move(kv)); // Move to avoid copy
        }

        // Run language model
        auto languageModelOutput = languageModel.Run(Ort::RunOptions{nullptr},
            languageModelInputNames.data(), languageModelInputTensors.data(), languageModelInputTensors.size(),
            languageModelOutputNames.data(), languageModelOutputNames.size());

        // Output 0: Logits [Batch, Seq, Vocab]
        float* logitsRaw = languageModelOutput[0].GetTensorMutableData<float>();
        auto logitsShape = languageModelOutput[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocabSize = logitsShape[2];
        int64_t seqDim = logitsShape[1];
        
        float* lastTokenLogits = logitsRaw + 1*(seqDim-1) * vocabSize;

        applyRepetitionPenalty(lastTokenLogits, vocabSize, generatedTokens, repetitionPenalty);
        // ----------------------------------

        // Greedy Search (Argmax) - Find the token with the highest score after penalty
        int64_t bestTokenId = 0;
        float maxScore = -std::numeric_limits<float>::infinity();

        for (int64_t v = 0; v < vocabSize; v++) {
            if (lastTokenLogits[v] > maxScore) {
                maxScore = lastTokenLogits[v];
                bestTokenId = v;
            }
        }

        nextTokenId = bestTokenId;
        if (nextTokenId == STOP_SPEECH_TOKEN) {
            std::cout << "\nStop token reached at step " << i << std::endl;
            break;
        }
        generatedTokens.push_back(nextTokenId);
        
        // Update KV Cache for next iteration
        pastKeyValues.clear(); // Clear old
        for (size_t k = 1; k < languageModelOutput.size(); k++) {
            pastKeyValues.push_back(std::move(languageModelOutput[k]));
        }
        currentSeqLen++;
    }
    return generatedTokens;
}

std::vector<int16_t> ChatterBox::synthesizeSpeech(std::vector<int64_t> generatedTokens) {
    // Run audio decoder model
    std::vector<int64_t> speechTokens;
    speechTokens.insert(speechTokens.end(), promptToken.begin(), promptToken.end());    
    speechTokens.insert(speechTokens.end(), generatedTokens.begin()+1, generatedTokens.end());
    speechTokens.insert(speechTokens.end(), {4299, 4299, 4299}); // Add silence at the end
    std::vector<Ort::Value> conditionalDecoderInputTensors;
    std::vector<int64_t> speechTokensDim{1, static_cast<int64_t>(speechTokens.size())};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, speechTokens.data(), speechTokens.size(),
        speechTokensDim.data(), speechTokensDim.size()));
    std::vector<int64_t> speakerEmbeddingsDim{1, 192};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speakerEmbeddings.data(), speakerEmbeddings.size(),
        speakerEmbeddingsDim.data(), speakerEmbeddingsDim.size()));
    std::vector<int64_t> speakerFeaturesDim{1, 500, 80};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speakerFeatures.data(), speakerFeatures.size(),
        speakerFeaturesDim.data(), speakerFeaturesDim.size()));
    auto audioOutput = conditionalDecoder.Run(
        Ort::RunOptions{nullptr},
        conditionalDecoderInputNames.data(), conditionalDecoderInputTensors.data(), conditionalDecoderInputTensors.size(),
        conditionalDecoderOutputNames.data(), conditionalDecoderOutputNames.size());
    
    std::vector<int16_t> audioBuffer;
    const float *audioOutputData = audioOutput.front().GetTensorData<float>();

    std::vector<int64_t> audioOutputShape = audioOutput.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioOutputCount = audioOutputShape[audioOutputShape.size() - 1];
    audioBuffer.reserve(audioOutputCount);

    // Convert float audio to int16
    for (int64_t i = 0; i < audioOutputCount; i++) {
        int16_t intAudioValue = static_cast<int16_t>(
            std::clamp(audioOutputData[i] * MAX_WAV_VALUE,
                        static_cast<float>(std::numeric_limits<int16_t>::min()),
                        static_cast<float>(std::numeric_limits<int16_t>::max())));
        audioBuffer.push_back(intAudioValue);
    }
    return audioBuffer;
}

void ChatterBox::applyRepetitionPenalty(float* logits, int64_t vocabSize, const std::vector<int64_t>& generatedTokens, float penalty) {
    std::unordered_set<int64_t> seenTokens;
    
    for (auto id : generatedTokens) {
        seenTokens.insert(id);
    }

    for (int64_t id : seenTokens) {
        float& score = logits[id];

        if (score < 0) {
            score *= penalty;
        } else {
            score /= penalty;
        }
    }
}