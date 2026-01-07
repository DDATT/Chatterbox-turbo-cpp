#include <iostream>
#include <chrono>
#include <string>
#include "wavfile.hpp"
#include "chatterbox.h"

int main(){
    ChatterBox chatterbox("ModelDir", false);
    chatterbox.LoadStyle("StyleDir");
    std::string text = "Hello, welcome to the world of C++ programming!";
    // std::vector<int64_t> inputIds = chatterbox.TokenizeText(text);
    // Predefined tokenized input for testing because I haven't implement huggingface tokenizer yet
    std::vector<int64_t> inputIds{5812, 11, 326, 338, 20105, 0, 220, 50274, 21039, 6949, 11, 703, 389, 345, 1804, 1909, 30, 50256, 50256};

    std::vector<int64_t> generatedTokens = chatterbox.SynthesizeSpeechTokens(inputIds);
    std::vector<int16_t> audioBuffer = chatterbox.synthesizeSpeech(generatedTokens);
    std::ofstream audioFile("test.wav", std::ios::binary);
    writeWavHeader(24000, 2, 1, (int32_t)audioBuffer.size(), audioFile);
    audioFile.write((const char *)audioBuffer.data(), sizeof(int16_t) * audioBuffer.size());
    audioFile.close();
    return 0;
}