import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of Google's Gemma3n model - next-generation multimodal Gemma architecture
// Based on: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models/gemma3n.py

// MARK: - Configuration

public struct Gemma3nTextConfiguration: Codable, Sendable {
    // Core model architecture fields that actually exist
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: [Int]
    public let numAttentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let numKeyValueHeads: Int
    public let laurelRank: Int
    public let altupActiveIdx: Int
    public let padTokenId: Int
    public let altupNumInputs: Int
    public let altupCoefClip: Float
    public let altupCorrectScale: Bool
    public let hiddenSizePerLayerInput: Int
    public let ropeLocalBaseFreq: Float
    public let ropeTheta: Float
    public let queryPreAttnScalar: Int?
    public let slidingWindow: Int
    public let ropeScaling: String?
    public let activationSparsityPattern: [Float]
    public let finalLogitSoftcapping: Float
    public let numKvSharedLayers: Int
    public let maxPositionEmbeddings: Int
    public let layerTypes: [String]
    public let altupLrMultiplier: Float?
    public let hiddenActivation: String
    
    // Optional fields from original config
    public let attentionBias: Bool?
    public let attentionDropout: Float?
    public let initializerRange: Float?
    public let torchDtype: String?
    public let useCache: Bool?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case laurelRank = "laurel_rank"
        case altupActiveIdx = "altup_active_idx"
        case padTokenId = "pad_token_id"
        case altupNumInputs = "altup_num_inputs"
        case altupCoefClip = "altup_coef_clip"
        case altupCorrectScale = "altup_correct_scale"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTheta = "rope_theta"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case ropeScaling = "rope_scaling"
        case activationSparsityPattern = "activation_sparsity_pattern"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numKvSharedLayers = "num_kv_shared_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case altupLrMultiplier = "altup_lr_multiplier"
        case hiddenActivation = "hidden_activation"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case initializerRange = "initializer_range"
        case torchDtype = "torch_dtype"
        case useCache = "use_cache"
    }
}

public struct Gemma3nVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let architecture: String
    public let hiddenSize: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabOffset: Int
    public let doPooling: Bool
    public let numClasses: Int
    
    // Optional fields from original config
    public let initializerRange: Float?
    public let labelNames: [String]?
    public let modelArgs: String?
    public let torchDtype: String?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture = "architecture"
        case hiddenSize = "hidden_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
        case doPooling = "do_pooling"
        case numClasses = "num_classes"
        case initializerRange = "initializer_range"
        case labelNames = "label_names"
        case modelArgs = "model_args"
        case torchDtype = "torch_dtype"
    }
}

public struct Gemma3nAudioConfiguration: Codable, Sendable {
    // Fields that actually exist in the config.json
    public let modelType: String
    public let inputFeatSize: Int
    public let hiddenSize: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabOffset: Int
    public let gradientClipping: Float
    public let confAttentionChunkSize: Int
    public let confAttentionContextLeft: Int
    public let confAttentionContextRight: Int
    public let confAttentionLogitCap: Float
    public let confNumAttentionHeads: Int
    public let confNumHiddenLayers: Int
    public let confConvKernelSize: Int
    public let confPositionalBiasSize: Int?
    public let confReductionFactor: Int
    public let confResidualWeight: Float
    
    // Optional fields from original config
    public let sscpConvChannelSize: [Int]?
    public let sscpConvGroupNormEps: Float?
    public let sscpConvKernelSize: [[Int]]?
    public let sscpConvStrideSize: [[Int]]?
    public let torchDtype: String?
    
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
        case sscpConvChannelSize = "sscp_conv_channel_size"
        case sscpConvGroupNormEps = "sscp_conv_group_norm_eps"
        case sscpConvKernelSize = "sscp_conv_kernel_size"
        case sscpConvStrideSize = "sscp_conv_stride_size"
        case torchDtype = "torch_dtype"
    }
}

public struct Gemma3nConfiguration: Codable, Sendable {
    // Main config with proper nested structure
    public let modelType: String
    public let textConfig: Gemma3nTextConfiguration
    public let visionConfig: Gemma3nVisionConfiguration
    public let audioConfig: Gemma3nAudioConfiguration
    public let audioSoftTokensPerImage: Int
    public let audioTokenId: Int
    public let boaTokenId: Int
    public let boiTokenId: Int
    public let imageTokenId: Int
    public let eoaTokenId: Int
    public let eoiTokenId: Int
    public let eosTokenId: [Int]?
    public let initializerRange: Float
    public let padTokenId: Int?
    public let visionSoftTokensPerImage: Int?
    
    // Optional fields from original config
    public let architectures: [String]?
    public let torchDtype: String?
    public let transformersVersion: String?
    
    // Computed properties for compatibility
    public var vocabSize: Int { textConfig.vocabSize }
    public var hiddenSize: Int { textConfig.hiddenSize }
    public var visionSoftTokensPerImageComputed: Int { visionSoftTokensPerImage ?? 256 }
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
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
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case architectures = "architectures"
        case torchDtype = "torch_dtype"
        case transformersVersion = "transformers_version"
    }
}

// MARK: - Core Components

private class Gemma3nRMSNorm: Module, UnaryLayer {
    let weight: MLXArray?
    let eps: Float
    let scaleShift: Float
    let withScale: Bool

    public init(dimensions: Int, eps: Float = 1e-6, scaleShift: Float = 0.0, withScale: Bool = true) {
        self.eps = eps
        self.scaleShift = scaleShift
        self.withScale = withScale
        self.weight = withScale ? MLXArray.ones([dimensions]) : nil
    }

    private func norm(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = norm(x.asType(.float32))
        
        if withScale, let weight = weight {
            return output.asType(x.dtype) * (weight + scaleShift)
        }
        
        return output.asType(x.dtype)
    }
}

private class RMSNoScale: Module, UnaryLayer {
    let eps: Float

    public init(eps: Float = 1e-5) {
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    }
}

// MARK: - Vision Components

private class MobileNetV5Block: Module {
    let conv1: Conv2d
    let norm1: BatchNorm
    let activation1: SiLU
    let conv2: Conv2d
    let norm2: BatchNorm
    let activation2: SiLU
    let skipConnection: Bool
    
    init(inputChannels: Int, outputChannels: Int, stride: Int = 1) {
        self.conv1 = Conv2d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: IntOrPair(3), stride: IntOrPair(stride), padding: IntOrPair(1), bias: false)
        self.norm1 = BatchNorm(featureCount: outputChannels)
        self.activation1 = SiLU()
        self.conv2 = Conv2d(inputChannels: outputChannels, outputChannels: outputChannels, kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1), bias: false)
        self.norm2 = BatchNorm(featureCount: outputChannels)
        self.activation2 = SiLU()
        self.skipConnection = (stride == 1 && inputChannels == outputChannels)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        out = activation1(norm1(conv1(out)))
        out = norm2(conv2(out))
        
        if skipConnection {
            out = out + x
        }
        
        return activation2(out)
    }
}

fileprivate class Gemma3nVisionTower: Module {
    let projector: Linear
    
    init(config: Gemma3nVisionConfiguration) {
        // Since the actual model uses a pre-trained mobilenetv5_300m_enc architecture,
        // we'll use a simple projector for now. The actual vision processing would
        // happen in the underlying MobileNetV5 implementation that gets loaded from weights.
        self.projector = Linear(config.hiddenSize, config.hiddenSize, bias: false)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // For now, assume the input has already been processed by the vision encoder
        // and just apply the final projection
        return projector(x)
    }
}

// MARK: - Audio Components

private class ConformerBlock: Module {
    let selfAttention: MultiHeadAttention
    let feedForward: Linear
    let norm1: LayerNorm
    let norm2: LayerNorm
    
    init(hiddenSize: Int, numHeads: Int, ffnDim: Int) {
        self.selfAttention = MultiHeadAttention(dimensions: hiddenSize, numHeads: numHeads, queryInputDimensions: hiddenSize, keyInputDimensions: hiddenSize, valueInputDimensions: hiddenSize)
        self.feedForward = Linear(hiddenSize, ffnDim, bias: false)
        self.norm1 = LayerNorm(dimensions: hiddenSize)
        self.norm2 = LayerNorm(dimensions: hiddenSize)
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Self-attention with residual
        let attnOut = selfAttention(x, keys: x, values: x, mask: mask)
        let x1 = norm1(x + attnOut)
        
        // Feed-forward with residual
        let ffnOut = feedForward(x1)
        return norm2(x1 + ffnOut)
    }
}

fileprivate class Gemma3nAudioTower: Module {
    let embeddings: Linear
    let encoder: [ConformerBlock]
    let projector: Linear
    
    init(config: Gemma3nAudioConfiguration) {
        // Audio embeddings
        self.embeddings = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        
        // Conformer encoder blocks
        var blocks: [ConformerBlock] = []
        for _ in 0..<config.confNumHiddenLayers {
            blocks.append(ConformerBlock(
                hiddenSize: config.hiddenSize,
                numHeads: config.confNumAttentionHeads,
                ffnDim: config.hiddenSize * 4
            ))
        }
        self.encoder = blocks
        
        // Feature projector
        self.projector = Linear(config.hiddenSize, config.hiddenSize, bias: false)
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Apply embeddings
        var hidden = embeddings(x)
        
        // Apply encoder blocks
        for block in encoder {
            hidden = block(hidden, mask: mask)
        }
        
        // Project features
        hidden = projector(hidden)
        
        return hidden
    }
}

// MARK: - Multimodal Model

public class Gemma3n: Module, LanguageModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    
    let config: Gemma3nConfiguration
    @ModuleInfo(key: "language_model") private var languageModel: Gemma3nLanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma3nVisionTower
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma3nAudioTower
    
    public init(_ config: Gemma3nConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.textConfig.numKeyValueHeads, count: config.textConfig.numHiddenLayers)
        
        // Initialize language model with config
        self._languageModel.wrappedValue = Gemma3nLanguageModel(config.textConfig)
        
        // Initialize vision tower with config
        self._visionTower.wrappedValue = Gemma3nVisionTower(config: config.visionConfig)
        
        // Initialize audio tower with config
        self._audioTower.wrappedValue = Gemma3nAudioTower(config: config.audioConfig)
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache)
    }
    
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        // Default implementation for LanguageModel protocol  
        return .tokens(input.text)
    }
    
    public func callAsFunction(
        inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        audioValues: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        // For now, just use the language model directly
        // Multimodal integration would require more complex implementation
        return languageModel(inputIds, cache: cache)
    }
    
    private func combineTextAndVision(textEmbeddings: MLXArray, visionEmbeddings: MLXArray) -> MLXArray {
        // Simple concatenation strategy - can be enhanced with more sophisticated fusion
        let batchSize = textEmbeddings.shape[0]
        let visionExpanded = MLX.broadcast(visionEmbeddings.expandedDimensions(axis: 1), to: [batchSize, config.visionSoftTokensPerImageComputed, visionEmbeddings.shape[1]])
        return MLX.concatenated([visionExpanded, textEmbeddings], axis: 1)
    }
    
    private func combineTextAndAudio(textEmbeddings: MLXArray, audioEmbeddings: MLXArray) -> MLXArray {
        // Simple concatenation strategy - can be enhanced with more sophisticated fusion
        let batchSize = textEmbeddings.shape[0]
        let audioExpanded = MLX.broadcast(audioEmbeddings.expandedDimensions(axis: 1), to: [batchSize, config.audioSoftTokensPerImage, audioEmbeddings.shape[1]])
        return MLX.concatenated([audioExpanded, textEmbeddings], axis: 1)
    }
}

// MARK: - Language Model Implementation

fileprivate class Gemma3nLanguageModel: Module, KVCacheDimensionProvider {
    public let kvHeads: [Int]
    
    @ModuleInfo var model: Gemma3nModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    
    let config: Gemma3nTextConfiguration
    
    public init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        
        self._model.wrappedValue = Gemma3nModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hiddenStates = model(inputs, cache: cache)
        return lmHead(hiddenStates)
    }
}

fileprivate class Gemma3nModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma3nDecoderLayer]
    @ModuleInfo var norm: Gemma3nRMSNorm
    
    let config: Gemma3nTextConfiguration
    
    init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        
        var layers: [Gemma3nDecoderLayer] = []
        for _ in 0..<config.numHiddenLayers {
            layers.append(Gemma3nDecoderLayer(config: config))
        }
        self._layers.wrappedValue = layers
        
        self._norm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            hidden = layer(hidden, cache: layerCache)
        }
        
        hidden = norm(hidden)
        return hidden
    }
}

// Simplified decoder layer implementation
fileprivate class Gemma3nDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3nAttention
    @ModuleInfo var mlp: Gemma3nMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma3nRMSNorm
    
    init(config: Gemma3nTextConfiguration) {
        self._selfAttn.wrappedValue = Gemma3nAttention(config: config)
        self._mlp.wrappedValue = Gemma3nMLP(config: config)
        self._inputLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let normed = inputLayernorm(x)
        let attnOutput = selfAttn(normed, cache: cache)
        let h = x + attnOutput
        
        let normed2 = postAttentionLayernorm(h)
        let mlpOutput = mlp(normed2)
        
        return h + mlpOutput
    }
}

// Simplified attention implementation
fileprivate class Gemma3nAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    
    let config: Gemma3nTextConfiguration
    
    init(config: Gemma3nTextConfiguration) {
        self.config = config
        self._queryProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        self._keyProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._valueProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
    }
    
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        // Simplified attention - would need full implementation for production
        let _ = queryProj(x)
        let _ = keyProj(x)
        let v = valueProj(x)
        
        // Simple scaled dot-product attention (simplified)
        let output = outputProj(v) // Placeholder implementation
        return output
    }
}

// Simplified MLP implementation
fileprivate class Gemma3nMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    
    let config: Gemma3nTextConfiguration
    
    init(config: Gemma3nTextConfiguration) {
        self.config = config
        let hiddenDim = config.intermediateSize[0]
        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDim, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDim, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDim, config.hiddenSize, bias: false)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        let activated = gate * sigmoid(gate) // SiLU activation
        return downProj(activated * up)
    }
}

// MARK: - Processor Configuration

public struct Gemma3nProcessorConfiguration: Codable, Sendable {
    // Fields from the preprocessor_config.json
    public let processorClass: String
    public let imageProcessorType: String
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let resample: Int
    public let rescaleFactor: Float
    public let size: ImageSize
    
    // Audio processing fields
    public let audioProcessorType: String?
    public let sampleRate: Int?
    public let audioSeqLength: Int?
    
    // Optional fields
    public let doConvertRgb: Bool?
    
    public var imageSize: Int {
        return size.height
    }
    
    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessorType = "image_processor_type"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case doResize = "do_resize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case resample = "resample"
        case rescaleFactor = "rescale_factor"
        case size = "size"
        case audioProcessorType = "audio_processor_type"
        case sampleRate = "sample_rate"
        case audioSeqLength = "audio_seq_length"
        case doConvertRgb = "do_convert_rgb"
    }
}

public struct ImageSize: Codable, Sendable {
    public let height: Int
    public let width: Int
}

// MARK: - Processor Implementation

public class Gemma3nProcessor: UserInputProcessor {
    private let config: Gemma3nProcessorConfiguration
    private let tokenizer: any Tokenizer
    
    public init(_ config: Gemma3nProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    
    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // Simplified preprocessing for now
        guard !images.isEmpty else {
            throw VLMError.imageProcessingFailure("No images to process")
        }
        
        // Create dummy processed image tensor
        let processedImage = MLXArray.zeros([images.count, 3, config.imageSize, config.imageSize])
        
        return (
            processedImage,
            THW(1, config.imageSize, config.imageSize)
        )
    }
    
    public func prepare(input: UserInput) async throws -> LMInput {
        let promptText = switch input.prompt {
        case .text(let text): text
        case .messages(let messages): messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        case .chat(let chatMessages): chatMessages.map(\.content).joined(separator: "\n")
        }
        
        // Handle images if present
        var imageTokens: [Int] = []
        if !input.images.isEmpty {
            // Convert UserInput.Image to CIImage array
            let ciImages = try input.images.map { imageInput in
                switch imageInput {
                case .url(let url):
                    guard let image = CIImage(contentsOf: url) else {
                        throw VLMError.imageProcessingFailure("Cannot load image from URL")
                    }
                    return image
                // For now, only handle URL case since .data is not available
                default:
                    throw VLMError.imageProcessingFailure("Only URL-based images are supported currently")
                }
            }
            
            let (_, _) = try preprocess(images: ciImages, processing: input.processing)
            // Add image tokens to the prompt (simplified approach)
            imageTokens = Array(repeating: 128256, count: config.imageSeqLength) // Placeholder tokens
        }
        
        // Handle audio if present (simplified)
        var audioTokens: [Int] = []
        if !input.videos.isEmpty { // Using videos array for audio for now
            audioTokens = Array(repeating: 128257, count: config.audioSeqLength ?? 1024) // Placeholder tokens
        }
        
        // Tokenize text
        let textTokens = tokenizer.encode(text: promptText)
        
        // Combine all tokens
        let allTokens = imageTokens + audioTokens + textTokens
        
        return LMInput(tokens: MLXArray(allTokens))
    }
}