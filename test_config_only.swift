#!/usr/bin/env swift

import Foundation

// Test script to validate Gemma3n configuration parsing without loading the full model
// This tests just the JSON decoding part

struct TestGemma3nConfiguration: Codable, Sendable {
    // Main config fields (flattened structure like actual config.json)
    let modelType: String
    let audioConfig: TestAudioConfiguration
    let audioSoftTokensPerImage: Int
    let audioTokenId: Int
    let boaTokenId: Int
    let boiTokenId: Int
    let imageTokenId: Int
    let eoaTokenId: Int
    let eoiTokenId: Int
    let eosTokenId: [Int]?
    let initializerRange: Float
    let padTokenId: Int?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case audioTokenId = "audio_token_id"
        case boaTokenId = "boa_token_id"
        case boiTokenId = "boi_token_id"
        case imageTokenId = "image_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoiTokenId = "eoi_token_id"
        case eosTokenId = "eos_token_id"
        case initializerRange = "initializer_range"
        case padTokenId = "pad_token_id"
    }
}

struct TestAudioConfiguration: Codable, Sendable {
    // Fields that actually exist in the config.json
    let modelType: String
    let inputFeatSize: Int
    let hiddenSize: Int
    let rmsNormEps: Float
    let vocabSize: Int
    let vocabOffset: Int
    let gradientClipping: Float
    let confAttentionChunkSize: Int
    let confAttentionContextLeft: Int
    let confAttentionContextRight: Int
    let confAttentionLogitCap: Float
    let confNumAttentionHeads: Int
    let confNumHiddenLayers: Int
    let confConvKernelSize: Int
    let confPositionalBiasSize: Int
    let confReductionFactor: Int
    let confResidualWeight: Float
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case inputFeatSize = "input_feat_size"
        case hiddenSize = "hidden_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
        case gradientClipping = "gradient_clipping"
        case confAttentionChunkSize = "conf_attention_chunk_size"
        case confAttentionContextLeft = "conf_attention_context_left"
        case confAttentionContextRight = "conf_attention_context_right"
        case confAttentionLogitCap = "conf_attention_logit_cap"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confConvKernelSize = "conf_conv_kernel_size"
        case confPositionalBiasSize = "conf_positional_bias_size"
        case confReductionFactor = "conf_reduction_factor"
        case confResidualWeight = "conf_residual_weight"
    }
}

func testConfigParsing() async {
    print("🧪 Testing Gemma3n configuration parsing...")
    
    let url = "https://huggingface.co/mlx-community/gemma-3n-E2B-it-4bit/raw/main/config.json"
    
    guard let configURL = URL(string: url) else {
        print("❌ Invalid URL")
        return
    }
    
    do {
        print("📥 Downloading config.json...")
        let (data, _) = try await URLSession.shared.data(from: configURL)
        
        print("📋 Parsing configuration...")
        let config = try JSONDecoder().decode(TestGemma3nConfiguration.self, from: data)
        
        print("✅ Successfully parsed Gemma3n configuration!")
        print("🏷️  Model type: \(config.modelType)")
        print("🎵 Audio token ID: \(config.audioTokenId)")
        print("🖼️  Image token ID: \(config.imageTokenId)")
        print("🔊 Audio config model type: \(config.audioConfig.modelType)")
        print("📏 Audio hidden size: \(config.audioConfig.hiddenSize)")
        print("🎯 Audio vocab size: \(config.audioConfig.vocabSize)")
        
    } catch {
        print("❌ Failed to parse configuration:")
        print("   Error: \(error)")
        
        if let decodingError = error as? DecodingError {
            switch decodingError {
            case .keyNotFound(let key, let context):
                print("   Missing key: \(key.stringValue)")
                print("   Context: \(context.debugDescription)")
            case .typeMismatch(let type, let context):
                print("   Type mismatch: expected \(type)")
                print("   Context: \(context.debugDescription)")
            case .valueNotFound(let type, let context):
                print("   Value not found: \(type)")
                print("   Context: \(context.debugDescription)")
            case .dataCorrupted(let context):
                print("   Data corrupted: \(context.debugDescription)")
            @unknown default:
                print("   Unknown decoding error")
            }
        }
    }
}

func main() async {
    await testConfigParsing()
}

await main()