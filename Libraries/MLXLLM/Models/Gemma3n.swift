// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// Port of Google's Gemma3n model - next-generation Gemma architecture
// Based on: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models/gemma3n.py

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

private class Gemma3nLaurelBlock: Module {
    @ModuleInfo(key: "linear_left") var linearLeft: Linear
    @ModuleInfo(key: "linear_right") var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") var postLaurelNorm: RMSNorm

    public init(_ config: Gemma3nTextConfiguration) {
        self._linearLeft.wrappedValue = Linear(config.hiddenSize, config.laurelRank, bias: false)
        self._linearRight.wrappedValue = Linear(config.laurelRank, config.hiddenSize, bias: false)
        self._postLaurelNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let laurelX = linearRight(linearLeft(x))
        let normedLaurelX = postLaurelNorm(laurelX)
        return x + normedLaurelX
    }
}

private class Gemma3nAttention: Module {
    let isSliding: Bool
    let nHeads: Int
    let nKvHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isKvSharedLayer: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let vNorm: RMSNoScale
    let rope: RoPE

    public init(_ config: Gemma3nTextConfiguration, layerIdx: Int, isKvSharedLayer: Bool) {
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self.nHeads = config.attentionHeads
        self.nKvHeads = config.kvHeads
        self.repeats = config.attentionHeads / config.kvHeads
        self.headDim = config.headDimensions
        self.layerIdx = layerIdx
        self.scale = 1.0
        self.isKvSharedLayer = isKvSharedLayer

        let dim = config.hiddenSize
        self._qProj.wrappedValue = Linear(dim, config.attentionHeads * config.headDimensions, bias: false)
        self._kProj.wrappedValue = Linear(dim, config.kvHeads * config.headDimensions, bias: false)
        self._vProj.wrappedValue = Linear(dim, config.kvHeads * config.headDimensions, bias: false)
        self._oProj.wrappedValue = Linear(config.attentionHeads * config.headDimensions, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDimensions, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDimensions, eps: config.rmsNormEps)
        self.vNorm = RMSNoScale(eps: config.rmsNormEps)

        let base = isSliding ? config.ropeLocalBaseFreq : config.ropeTheta
        self.rope = RoPE(dimensions: config.headDimensions, traditional: false, base: base)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]

        var queries = qProj(x)
        queries = queries.reshaped([B, L, -1, headDim])
        queries = qNorm(queries)

        var offset = 0
        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer && cache != nil {
            // For shared layers, retrieve KV from the designated cache layer
            let cacheState = cache!.state
            keys = cacheState[0]
            values = cacheState[1]
            offset = cache!.offset
        } else {
            if let cache = cache {
                offset = cache.offset
            }

            keys = kProj(x).reshaped([B, L, -1, headDim])
            keys = kNorm(keys)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: offset)

            values = vProj(x).reshaped([B, L, -1, headDim])
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)

            if let cache = cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        let reshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return oProj(reshaped)
    }
}

private func geluTopk(inputs: MLXArray, stdMultiplier: MLXArray) -> MLXArray {
    let inputsMean = inputs.mean(axis: -1, keepDims: true)
    let variance = inputs.variance(axis: -1, keepDims: true)
    let inputsStd = sqrt(variance)
    let cutoffX = inputsMean + inputsStd * stdMultiplier.asType(inputsStd.dtype)
    return gelu(maximum(MLXArray(0), inputs - cutoffX))
}

private class MLP: Module {
    let hiddenSize: Int
    let intermediateSize: Int
    let activationSparsity: Float
    private let stdMultiplier: MLXArray?

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: Gemma3nTextConfiguration, layerIdx: Int = 0) {
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize[layerIdx]
        
        if let sparsityPattern = config.activationSparsityPattern {
            self.activationSparsity = sparsityPattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }

        if activationSparsity > 0 {
            // Approximate erfInv for now - this would need to be implemented in MLX Swift
            let x = 2 * activationSparsity - 1
            let erfInvApprox = 0.8862269254527579 * x  // Simplified approximation
            self.stdMultiplier = MLXArray(sqrt(2.0) * erfInvApprox)
        } else {
            self.stdMultiplier = nil
        }

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateProj = self.gateProj(x)
        
        let activations: MLXArray
        if activationSparsity > 0.0, let stdMultiplier = stdMultiplier {
            activations = geluTopk(inputs: gateProj, stdMultiplier: stdMultiplier)
        } else {
            activations = gelu(gateProj)
        }
        
        let upProj = self.upProj(x)
        let downProj = self.downProj(activations * upProj)
        return downProj
    }
}

private class Gemma3nAltUp: Module {
    @ParameterInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: RMSNorm
    
    let config: Gemma3nTextConfiguration

    public init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        
        self._correctOutputScale.wrappedValue = MLXArray.zeros([config.hiddenSize])
        self._correctionCoefs.wrappedValue = Linear(config.altupNumInputs, config.altupNumInputs, bias: false)
        self._predictionCoefs.wrappedValue = Linear(config.altupNumInputs, config.altupNumInputs * config.altupNumInputs, bias: false)
        self._modalityRouter.wrappedValue = Linear(config.hiddenSize, config.altupNumInputs, bias: false)
        self._routerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    private func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        let routerInputs = routerNorm(x) * pow(Float(config.hiddenSize), -1.0)
        let routed = modalityRouter(routerInputs).asType(.float32)
        return tanh(routed)
    }

    public func predict(_ x: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(x[config.altupActiveIdx])
        
        var predictionWeight = predictionCoefs.weight.asType(.float32)
        
        if let clipValue = config.altupCoefClip {
            predictionWeight = clip(predictionWeight, min: -clipValue, max: clipValue)
        }
        
        let allCoefs = predictionCoefs(modalities)
            .reshaped([modalities.shape[0], modalities.shape[1], config.altupNumInputs, config.altupNumInputs])
            .transposed(0, 1, 3, 2)
        
        let xUp = x.asType(.float32)
        let xPermuted = xUp.transposed(1, 2, 3, 0)
        let predictions = matmul(xPermuted, allCoefs)
        let predictionsTransposed = predictions.transposed(3, 0, 1, 2)
        let result = predictionsTransposed + xUp
        
        return result.asType(x.dtype)
    }

    public func correct(_ predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)
        
        var correctionWeight = correctionCoefs.weight.asType(.float32)
        
        if let clipValue = config.altupCoefClip {
            correctionWeight = clip(correctionWeight, min: -clipValue, max: clipValue)
        }
        
        let allCoefs = correctionCoefs(modalities) + 1.0
        
        let activeX = predictions[config.altupActiveIdx]
        let innovation = activated - activeX
        
        let allCoefsTransposed = allCoefs.transposed(2, 1, 0)
        let corrected = innovation.expandedDimensions(axis: 0) * allCoefsTransposed.expandedDimensions(axis: 1)
        let result = corrected + predictions
        
        return result.asType(activated.dtype)
    }
}

private class Gemma3nDecoderLayer: Module {
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let layerIdx: Int
    let isSliding: Bool
    let slidingWindow: Int
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3nAttention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "altup") var altup: Gemma3nAltUp
    @ModuleInfo(key: "laurel") var laurel: Gemma3nLaurelBlock
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm

    public init(_ config: Gemma3nTextConfiguration, layerIdx: Int, isKvSharedLayer: Bool) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        self.slidingWindow = config.slidingWindow
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma3nAttention(config, layerIdx: layerIdx, isKvSharedLayer: isKvSharedLayer)
        self._mlp.wrappedValue = MLP(config, layerIdx: layerIdx)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        self._altup.wrappedValue = Gemma3nAltUp(config)
        self._laurel.wrappedValue = Gemma3nLaurelBlock(config)
        self._perLayerInputGate.wrappedValue = Linear(hiddenSize, hiddenSizePerLayerInput, bias: false)
        self._perLayerProjection.wrappedValue = Linear(hiddenSizePerLayerInput, hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        let predictions = altup.predict(x)
        let activePrediction = predictions[config.altupActiveIdx]

        let activePredictionNormed = inputLayernorm(activePrediction)
        let laurelOutput = laurel(activePredictionNormed)

        let attn = selfAttn(activePredictionNormed, mask: mask, cache: cache)
        let attnNormed = postAttentionLayernorm(attn)

        let attnGated = activePrediction + attnNormed
        let attnLaurel = (attnGated + laurelOutput) * pow(2.0, -0.5)

        let attnNorm = preFeedforwardLayernorm(attnLaurel)
        let attnFfw = mlp(attnNorm)
        let attnFfwNorm = postFeedforwardLayernorm(attnFfw)
        let attnFfwLaurelGated = attnLaurel + attnFfwNorm

        var correctedPredictions = altup.correct(predictions, activated: attnFfwLaurelGated)

        var firstPrediction = correctedPredictions[config.altupActiveIdx]
        if config.altupCorrectScale {
            firstPrediction = firstPrediction * altup.correctOutputScale
        }

        firstPrediction = perLayerInputGate(firstPrediction)
        firstPrediction = gelu(firstPrediction)

        if let perLayerInput = perLayerInput {
            firstPrediction = firstPrediction * perLayerInput
        }

        firstPrediction = perLayerProjection(firstPrediction)
        firstPrediction = postPerLayerInputNorm(firstPrediction)

        // Update the corrected predictions
        for i in 1..<correctedPredictions.shape[0] {
            correctedPredictions[i] = correctedPredictions[i] + firstPrediction
        }

        return correctedPredictions
    }
}

private func logitSoftcap(softcap: Float, x: MLXArray) -> MLXArray {
    let out = tanh(x / softcap)
    return out * softcap
}

private class Gemma3nModelInner: Module {
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let hiddenSizePerLayerInput: Int
    let vocabularySize: Int
    let vocabularySizePerLayerInput: Int
    let numHiddenLayers: Int
    let firstKvSharedLayerIdx: Int
    let firstSlidingIdx: Int
    let firstFullIdx: Int
    let layerIdxToCacheIdx: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [Gemma3nDecoderLayer]
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm
    let altupProjections: [Linear]
    let altupUnembedProjections: [Linear]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.vocabularySize = config.vocabularySize
        self.vocabularySizePerLayerInput = config.vocabularySizePerLayerInput
        self.numHiddenLayers = config.hiddenLayers
        self.firstKvSharedLayerIdx = config.hiddenLayers - config.numKvSharedLayers

        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        
        // Create layers
        var layerArray: [Gemma3nDecoderLayer] = []
        for layerIdx in 0..<config.hiddenLayers {
            let isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx
            layerArray.append(Gemma3nDecoderLayer(config, layerIdx: layerIdx, isKvSharedLayer: isKvSharedLayer))
        }
        self.layers = layerArray

        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: config.vocabularySizePerLayerInput,
            dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
        )

        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenLayers * config.hiddenSizePerLayerInput,
            bias: false
        )

        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSizePerLayerInput,
            eps: config.rmsNormEps
        )

        // Create altup projections
        var altupProjArray: [Linear] = []
        var altupUnembedProjArray: [Linear] = []
        for _ in 1..<config.altupNumInputs {
            altupProjArray.append(Linear(config.hiddenSize, config.hiddenSize, bias: false))
            altupUnembedProjArray.append(Linear(config.hiddenSize, config.hiddenSize, bias: false))
        }
        self.altupProjections = altupProjArray
        self.altupUnembedProjections = altupUnembedProjArray

        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Find indices for layer types
        self.firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.firstFullIdx = config.layerTypes.firstIndex(of: "full_attention") ?? 0

        // Build layer index to cache index mapping
        let concreteLayerTypes = Array(config.layerTypes.prefix(firstKvSharedLayerIdx))
        let sharedFullIdx = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0

        var layerIdxToCacheIdxArray: [Int] = []
        for (i, layerType) in config.layerTypes.enumerated() {
            if i < firstKvSharedLayerIdx {
                layerIdxToCacheIdxArray.append(i)
            } else {
                if layerType == "full_attention" {
                    layerIdxToCacheIdxArray.append(sharedFullIdx)
                } else if layerType == "sliding_attention" {
                    layerIdxToCacheIdxArray.append(sharedSlidingIdx)
                } else {
                    fatalError("Unknown layer type: \(layerType)")
                }
            }
        }
        self.layerIdxToCacheIdx = layerIdxToCacheIdxArray
    }

    public func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        let perLayerInputsMask = inputIds .< vocabularySizePerLayerInput
        let tokens = MLX.where(perLayerInputsMask, inputIds, MLXArray.zeros(like: inputIds))
        let result = embedTokensPerLayer(tokens) * pow(Float(hiddenSizePerLayerInput), 0.5)
        
        let newShape = Array(inputIds.shape) + [numHiddenLayers, hiddenSizePerLayerInput]
        return result.reshaped(newShape)
    }

    public func projectPerLayerInputs(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray) -> MLXArray {
        let perLayerProjection = perLayerModelProjection(inputsEmbeds) * pow(Float(hiddenSize), -0.5)
        
        let newShape = Array(inputsEmbeds.shape.dropLast()) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        let projectionReshaped = perLayerProjection.reshaped(newShape)
        let projectionNormed = perLayerProjectionNorm(projectionReshaped)
        
        return (projectionNormed + perLayerInputs) * pow(2.0, -0.5)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs) * pow(Float(hiddenSize), 0.5)
        let perLayerInputs = getPerLayerInputs(inputs)
        let finalPerLayerInputs = projectPerLayerInputs(h, perLayerInputs: perLayerInputs)

        let cacheToUse: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let mask = createAttentionMask(h: h, cache: cache)

        let h0 = h
        let targetMagnitude = sqrt(h0.square().mean(axis: -1, keepDims: true))

        var hList = [h0]
        for proj in altupProjections {
            hList.append(proj(h0))
        }
        
        var hStacked = stacked(hList, axis: 0)
        let mags = sqrt(hStacked[1...].square().mean(axis: -1, keepDims: true))
        let minValue = MLXArray(Float.leastNormalMagnitude).asType(h0.dtype)
        let ratios = targetMagnitude / maximum(mags, minValue)
        for i in 1..<hStacked.shape[0] {
            hStacked[i] = hStacked[i] * ratios
        }

        for (i, layer) in layers.enumerated() {
            let perLayerInput = finalPerLayerInputs[0..., 0..., i, 0...]
            let cacheIdx = layerIdxToCacheIdx[i]
            
            hStacked = layer(hStacked, mask: mask, cache: cacheToUse[cacheIdx], perLayerInput: perLayerInput)
        }

        // Per-layer inputs to single output
        let targetMagnitudeFinal = sqrt(hStacked[0].square().mean(axis: -1, keepDims: true))
        for (i, proj) in altupUnembedProjections.enumerated() {
            hStacked[i + 1] = proj(hStacked[i + 1])
        }
        
        let magsFinal = sqrt(hStacked[1...].square().mean(axis: -1, keepDims: true))
        let ratiosFinal = targetMagnitudeFinal / maximum(magsFinal, minValue)
        for i in 1..<hStacked.shape[0] {
            hStacked[i] = hStacked[i] * ratiosFinal
        }

        let hMean = hStacked.mean(axis: 0)
        return norm(hMean)
    }
}

public class Gemma3nModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    
    let modelType: String
    private let model: Gemma3nModelInner
    private let finalLogitSoftcapping: Float?

    public init(_ config: Gemma3nConfiguration) {
        self.modelType = config.textConfig.modelType
        self.vocabularySize = config.textConfig.vocabularySize
        self.kvHeads = Array(repeating: config.textConfig.kvHeads, count: config.textConfig.hiddenLayers)
        self.finalLogitSoftcapping = config.textConfig.finalLogitSoftcapping
        self.model = Gemma3nModelInner(config.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        var logits = model.embedTokens.asLinear(out)
        
        if let softcapping = finalLogitSoftcapping {
            logits = logitSoftcap(softcap: softcapping, x: logits)
        }
        
        return logits
    }


    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        
        for (k, v) in weights {
            var newKey = k
            
            // Handle model.language_model prefix mapping for text model weights
            if newKey.hasPrefix("model.language_model.") {
                newKey = String(newKey.dropFirst(21)) // Remove "model.language_model." prefix
            }
            
            // Skip rotary embedding frequencies and missing parameters
            if newKey.contains("rotary_emb.inv_freq") ||
               newKey.contains("correction_coefs") ||
               newKey.contains("prediction_coefs") ||
               newKey.contains("correct_output_scale") ||
               newKey.contains("q_norm") ||
               newKey.contains("k_norm") ||
               newKey.contains("router_norm") ||
               newKey.contains("post_laurel_norm") ||
               newKey.contains("post_per_layer_input_norm") ||
               newKey.contains("per_layer_model_projection") ||
               newKey.contains("per_layer_projection_norm") ||
               newKey.contains("per_layer_input_gate") ||
               newKey.contains("per_layer_projection") ||
               newKey.contains("pre_feedforward_layernorm") {
                continue
            }
            
            sanitizedWeights[newKey] = v
        }
        
        return sanitizedWeights
    }

    public var layers: [Module] {
        return model.layers.map { $0 as Module }
    }

    public var headDim: Int {
        return model.config.headDimensions
    }

    public var nKVHeads: Int {
        return model.config.kvHeads
    }

    public func makeCache() -> [KVCache] {
        var caches: [KVCache] = []
        let firstKvSharedLayerIdx = model.config.hiddenLayers - model.config.numKvSharedLayers
        
        for (index, layerType) in model.config.layerTypes.enumerated() {
            if index >= firstKvSharedLayerIdx {
                break
            }
            
            if layerType == "full_attention" {
                caches.append(KVCacheSimple())
            } else if layerType == "sliding_attention" {
                caches.append(RotatingKVCache(maxSize: model.config.slidingWindow, keep: 0))
            } else {
                fatalError("Unknown layer type: \(layerType)")
            }
        }
        
        return caches
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        NoSystemMessageGenerator()
    }
}

// MARK: - Configuration

public struct Gemma3nTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: [Int]
    public let attentionHeads: Int
    public let headDimensions: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let kvHeads: Int
    public let laurelRank: Int
    public let altupActiveIdx: Int
    public let altupNumInputs: Int
    private let _altupCoefClip: Float?
    public var altupCoefClip: Float? { _altupCoefClip }
    public let altupCorrectScale: Bool
    public let hiddenSizePerLayerInput: Int
    public let ropeLocalBaseFreq: Float
    public let ropeTheta: Float
    public let queryPreAttnScalar: Float
    public let slidingWindow: Int
    public let maxPositionEmbeddings: Int
    public let layerTypes: [String]
    public let activationSparsityPattern: [Float]?
    public let finalLogitSoftcapping: Float
    public let numKvSharedLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case kvHeads = "num_key_value_heads"
        case laurelRank = "laurel_rank"
        case altupActiveIdx = "altup_active_idx"
        case altupNumInputs = "altup_num_inputs"
        case _altupCoefClip = "altup_coef_clip"
        case altupCorrectScale = "altup_correct_scale"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTheta = "rope_theta"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case activationSparsityPattern = "activation_sparsity_pattern"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case numKvSharedLayers = "num_kv_shared_layers"
    }
}

public struct Gemma3nConfiguration: Codable, Sendable {
    public let textConfig: Gemma3nTextConfiguration
    
    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}

// MARK: - LoRA Support

extension Gemma3nModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.selfAttn, ["q_proj", "v_proj", "k_proj", "o_proj"]) }
    }
}