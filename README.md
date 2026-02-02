# Chatterbox-turbo-cpp

A C++ implementation of Chatterbox-turbo inference using ONNX Runtime for text-to-speech synthesis.

## Overview

Chatterbox-turbo is a family of three state-of-the-art, open-source text-to-speech models by Resemble AI. It leverages ONNX Runtime for fast inference and supports both CPU and CUDA acceleration.

## Features

- **High Performance**: Optimized C++ implementation for fast inference
- **ONNX Runtime Integration**: Cross-platform model deployment
- **Style Transfer**: Support for custom voice styles and speaker embeddings

## Architecture

This implementation uses a hybrid approach to optimize inference performance while maintaining consistency with the Python reference implementation:

- **Audio Embeddings**: To optimize C++ inference and ensure results match the Python implementation, audio embeddings are pre-computed in Python and stored as `.bin` files rather than running the embedding model in C++.
- **Tokenizer**: The HuggingFace tokenizer has been reimplemented in C++ with assistance from Claude Sonnet 4.5. While not exhaustively tested, it performs well across all test cases.
- **ONNX Models**: The following three models are loaded and executed in C++:
  - **embed_tokens.onnx**: Text token embedding layer
  - **language_model.onnx**: Language model with attention mechanism and KV caching
  - **conditional_decoder.onnx**: Conditional audio decoder for speech synthesis

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler
- ONNX Runtime library 1.22 or higher
- CUDA Toolkit (optional, for GPU acceleration)

## Building

1. Clone the repository:
```bash
git clone https://github.com/DDATT/Chatterbox-turbo-cpp.git
cd Chatterbox-turbo-cpp
```

2. Ensure ONNX Runtime is installed and accessible:
   - Place ONNX Runtime headers in `onnxruntime/include/`
   - Place ONNX Runtime library in `onnxruntime/lib/`

3. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

## Project Structure

```
Chatterbox-turbo-cpp/
├── CMakeLists.txt          # Build configuration
├── main.cpp                # Entry point and usage example
├── README.md               # This file
├── LICENSE                 # License information
├── assets/
│   └── tokenizer.json      # BPE tokenizer configuration
├── include/
│   ├── chatterbox.h        # Main ChatterBox class header
│   ├── bpe_tokenizer.hpp   # BPE tokenizer header
│   ├── wavfile.hpp         # WAV file utilities
│   └── nlohmann/
│       └── json.hpp        # JSON parsing library
└── src/
    ├── chatterbox.cpp      # ChatterBox implementation
    └── bpe_tokenizer.cpp   # BPE tokenizer implementation
```

## Usage

### Basic Example

```cpp
#include "chatterbox.h"
#include "bpe_tokenizer.hpp"
#include "wavfile.hpp"

int main() {
    // Load tokenizer
    BPETokenizer tokenizer;
    if (!tokenizer.loadFromFile("assets/tokenizer.json")) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }
    
    // Initialize ChatterBox (set second parameter to true for CUDA)
    ChatterBox chatterbox("ModelDir", false);
    chatterbox.LoadStyle("StyleDir");

    // Synthesize speech
    std::string text = "Hello, welcome to my world!";
    std::vector<int64_t> inputIds = tokenizer.encode(text, true);
    std::vector<int64_t> generatedTokens = chatterbox.SynthesizeSpeechTokens(inputIds);
    std::vector<int16_t> audioBuffer = chatterbox.synthesizeSpeech(generatedTokens);
    
    // Write to WAV file
    std::ofstream audioFile("test.wav", std::ios::binary);
    writeWavHeader(24000, 2, 1, audioBuffer.size(), audioFile);
    audioFile.write((const char*)audioBuffer.data(), 
                    sizeof(int16_t) * audioBuffer.size());
    audioFile.close();
    
    return 0;
}
```

### Model Setup

1. **Model Directory**: Place your ONNX models in a directory (e.g., `ModelDir/`):
   - `embed_tokens.onnx`
   - `language_model.onnx`
   - `conditional_decoder.onnx`

2. **Style Directory**: Create a style directory (e.g., `StyleDir/`) with:
   - `cond_emb.bin` - Conditional embeddings
   - `prompt_token.bin` - Prompt tokens
   - `speaker_embeddings.bin` - Speaker embeddings
   - `speaker_features.bin` - Speaker features

### Configuration

You can adjust synthesis parameters:

```cpp
chatterbox.repetitionPenalty = 1.2f;  // Control repetition (default: 1.2)
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Model based on [Chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
- Built with [ONNX Runtime](https://onnxruntime.ai/)
- Uses [nlohmann/json](https://github.com/nlohmann/json) for JSON parsing

## Contact

For questions or issues, feel free to open an issue on the [GitHub repository](https://github.com/DDATT/Chatterbox-turbo-cpp).
