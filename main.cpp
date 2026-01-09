#include <iostream>
#include <chrono>
#include <string>
#include "wavfile.hpp"
#include "chatterbox.h"
#include "bpe_tokenizer.hpp"
int main(){
    BPETokenizer tokenizer;
    std::cout << "\nLoading tokenizer from tokenizer.json..." << std::endl;
    if (!tokenizer.loadFromFile("assets/tokenizer.json")) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }
    
    ChatterBox chatterbox("ModelDir", false);
    chatterbox.LoadStyle("StyleDir");

    std::string text = "Hello, welcome to my world!";

    std::vector<int64_t> inputIds = tokenizer.encode(text, true);

    std::vector<int64_t> generatedTokens = chatterbox.SynthesizeSpeechTokens(inputIds);
    std::vector<int16_t> audioBuffer = chatterbox.synthesizeSpeech(generatedTokens);
    std::ofstream audioFile("test.wav", std::ios::binary);
    writeWavHeader(24000, 2, 1, (int32_t)audioBuffer.size(), audioFile);
    audioFile.write((const char *)audioBuffer.data(), sizeof(int16_t) * audioBuffer.size());
    audioFile.close();
    return 0;
}