#!/usr/bin/env swift

import Foundation

// Test creating a minimal Gemma3n configuration to see if it works
struct TestConfig: Codable {
    let modelType = "gemma3n"
    let audioConfig = TestAudioConfig()
    let audioSoftTokensPerImage = 188
    let audioTokenId = 262273
    let boaTokenId = 256000
    let boiTokenId = 255999
    let imageTokenId = 262145
    let eoaTokenId = 262272
    let eoiTokenId = 262144
    let eosTokenId: [Int]? = [1, 106]
    let initializerRange: Float = 0.02
    let padTokenId: Int? = nil
    
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

struct TestAudioConfig: Codable {
    let modelType = "gemma3n_audio"
    let inputFeatSize = 128
    let hiddenSize = 1536
    let rmsNormEps: Float = 0.000001
    let vocabSize = 128
    let vocabOffset = 262272
    let gradientClipping: Float = 10000000000.0
    let confAttentionChunkSize = 12
    let confAttentionContextLeft = 13
    let confAttentionContextRight = 0
    let confAttentionLogitCap: Float = 50.0
    let confNumAttentionHeads = 8
    let confNumHiddenLayers = 12
    let confConvKernelSize = 5
    let confPositionalBiasSize = 256
    let confReductionFactor = 4
    let confResidualWeight: Float = 0.5
    
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

func testConfigCreation() {
    print("🧪 Testing Gemma3n configuration creation...")
    
    do {
        let config = TestConfig()
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        let data = try encoder.encode(config)
        let json = String(data: data, encoding: .utf8) ?? "Failed to encode"
        
        print("✅ Successfully created test configuration:")
        print(json)
        
        // Test decoding
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(TestConfig.self, from: data)
        
        print("\n✅ Successfully decoded configuration:")
        print("  Model type: \(decoded.modelType)")
        print("  Audio token ID: \(decoded.audioTokenId)")
        print("  Audio config model type: \(decoded.audioConfig.modelType)")
        
    } catch {
        print("❌ Failed to create/encode configuration: \(error)")
    }
}

testConfigCreation()