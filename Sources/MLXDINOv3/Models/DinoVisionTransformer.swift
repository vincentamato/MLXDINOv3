//
//  DinoVisionTransformer.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public struct DinoOutput {
    public let lastHiddenState: MLXArray
    public let poolerOutput: MLXArray
    public let hiddenStates: [String: MLXArray]?
    public let attentions: [MLXArray]?

    public init(lastHiddenState: MLXArray, poolerOutput: MLXArray, hiddenStates: [String: MLXArray]? = nil, attentions: [MLXArray]? = nil) {
        self.lastHiddenState = lastHiddenState
        self.poolerOutput = poolerOutput
        self.hiddenStates = hiddenStates
        self.attentions = attentions
    }
}

public class DinoVisionTransformer: Module {
    @ModuleInfo public var embeddings: Embeddings
    public let ropeEmbed: RoPEPositionEncoding
    @ModuleInfo public var layer: [SelfAttentionBlock]
    @ModuleInfo public var norm: LayerNorm
    @ModuleInfo(key: "cls_norm") public var clsNorm: LayerNorm?
    public let embedDim: Int
    public let numHeads: Int
    public let patchSize: Int
    public let nBlocks: Int
    public let numRegisterTokens: Int
    let untieCLSAndPatchNorms: Bool

    public init(imgSize: Int,
                patchSize: Int,
                inChannels: Int,
                posEmbedRopeBase: Float,
                posEmbedRopeNormalizeCoords: String,
                posEmbedRopeRescale: Float?,
                posEmbedRopeDtype: String,
                embedDim: Int,
                depth: Int,
                numHeads: Int,
                ffnRatio: Float,
                queryBias: Bool,
                keyBias: Bool,
                valueBias: Bool,
                layerscaleInit: Float?,
                layerNormEps: Float,
                ffnBias: Bool,
                projBias: Bool,
                useGatedMLP: Bool,
                numRegisterTokens: Int,
                untieCLSAndPatchNorms: Bool)
    {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.patchSize = patchSize
        nBlocks = depth
        self.numRegisterTokens = numRegisterTokens
        self.untieCLSAndPatchNorms = untieCLSAndPatchNorms

        _embeddings.wrappedValue = Embeddings(imgSize: imgSize,
                                              patchSize: patchSize,
                                              inChannels: inChannels,
                                              embedDim: embedDim,
                                              layerNormEps: layerNormEps,
                                              numRegisterTokens: numRegisterTokens)

        let dtypeMap: [String: DType] = [
            "fp32": .float32,
            "fp16": .float16,
            "bf16": .bfloat16,
        ]
        let ropeDtype = dtypeMap[posEmbedRopeDtype.lowercased()] ?? .float32

        ropeEmbed = RoPEPositionEncoding(embedDim: embedDim,
                                         numHeads: numHeads,
                                         base: posEmbedRopeBase,
                                         normalizeCoords: posEmbedRopeNormalizeCoords,
                                         rescaleCoords: posEmbedRopeRescale,
                                         dtype: ropeDtype)

        var layerList: [SelfAttentionBlock] = []
        for _ in 0 ..< depth {
            layerList.append(
                SelfAttentionBlock(dim: embedDim,
                                   numHeads: numHeads,
                                   ffnRatio: ffnRatio,
                                   queryBias: queryBias,
                                   keyBias: keyBias,
                                   valueBias: valueBias,
                                   projBias: projBias,
                                   ffnBias: ffnBias,
                                   initValues: layerscaleInit,
                                   layerNormEps: layerNormEps,
                                   useGatedMLP: useGatedMLP)
            )
        }
        _layer.wrappedValue = layerList

        _norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: layerNormEps)

        if untieCLSAndPatchNorms {
            _clsNorm.wrappedValue = LayerNorm(dimensions: embedDim, eps: layerNormEps)
        } else {
            _clsNorm.wrappedValue = nil
        }
    }

    private func prepareTokensWithMasks(_ x: MLXArray,
                                        masks _: MLXArray? = nil) -> (MLXArray, Int, Int)
    {
        var x = embeddings.patch_embeddings(x)

        let (B, H, W, _) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        x = x.reshaped(B, H * W, x.dim(-1))

        let cls = embeddings.cls_token
        let clsExpanded = broadcast(cls, to: [B, cls.dim(1), cls.dim(2)])

        if numRegisterTokens > 0, let register = embeddings.register_tokens {
            let registerExpanded = broadcast(register, to: [B, register.dim(1), register.dim(2)])
            x = concatenated([clsExpanded, registerExpanded, x], axis: 1)
        } else {
            x = concatenated([clsExpanded, x], axis: 1)
        }

        return (x, H, W)
    }

    public func forwardFeatures(_ x: MLXArray,
                                masks: MLXArray? = nil,
                                outputHiddenStates: Bool = false,
                                outputAttentions: Bool = false) -> (features: [String: MLXArray], attentions: [MLXArray]?)
    {
        var (x, H, W) = prepareTokensWithMasks(x, masks: masks)

        var allHiddenStates: [MLXArray] = []
        var allAttentions: [MLXArray] = []

        if outputHiddenStates {
            allHiddenStates.append(x)
        }

        for block in layer {
            let rope = ropeEmbed(H: H, W: W)
            let (blockOutput, blockAttention) = block(x, rope: rope, returnAttentionWeights: outputAttentions)
            x = blockOutput

            if outputHiddenStates {
                allHiddenStates.append(x)
            }

            if outputAttentions, let attention = blockAttention {
                allAttentions.append(attention)
            }
        }

        let xNormClsReg: MLXArray
        let xNormPatch: MLXArray

        if untieCLSAndPatchNorms, let clsNorm {
            xNormClsReg = clsNorm(x[0..., ..<(numRegisterTokens + 1), 0...])
            xNormPatch = norm(x[0..., (numRegisterTokens + 1)..., 0...])
        } else {
            let xNorm = norm(x)
            xNormClsReg = xNorm[0..., ..<(numRegisterTokens + 1), 0...]
            xNormPatch = xNorm[0..., (numRegisterTokens + 1)..., 0...]
        }

        var features: [String: MLXArray] = [
            "x_norm_clstoken": xNormClsReg[0..., 0, 0...],
            "x_register_tokens": xNormClsReg[0..., 1 ..< (numRegisterTokens + 1), 0...],
            "x_norm_patchtokens": xNormPatch,
            "x_prenorm": x,
        ]

        if outputHiddenStates {
            for (i, hiddenState) in allHiddenStates.enumerated() {
                features["hidden_state_\(i)"] = hiddenState
            }
        }

        let attentions = outputAttentions ? allAttentions : nil
        return (features: features, attentions: attentions)
    }

    private func printArrayStats(_ name: String, _ arr: MLXArray) {
        let minVal = arr.min().item(Float.self)
        let maxVal = arr.max().item(Float.self)
        let meanVal = mean(arr).item(Float.self)
        let variance = mean((arr - meanVal) * (arr - meanVal))
        let stdVal = sqrt(variance).item(Float.self)
        let norm = sqrt((arr * arr).sum()).item(Float.self)

        print("   \(name):")
        print("     Shape: \(arr.shape)")
        print("     Range: [\(String(format: "%.6f", minVal)), \(String(format: "%.6f", maxVal))]")
        print("     Mean: \(String(format: "%.6f", meanVal))")
        print("     Std: \(String(format: "%.6f", stdVal))")
        print("     Norm: \(String(format: "%.6f", norm))")

        let flat = arr.reshaped(-1)
        let first10 = (0 ..< min(10, flat.size)).map { flat[$0].item(Float.self) }
        print("     First 10: \(first10.map { String(format: "%.6f", $0) }.joined(separator: ", "))")
    }

    public func getIntermediateLayers(_ x: MLXArray,
                                      n: Int = 1,
                                      indices: [Int]? = nil,
                                      reshape: Bool = false,
                                      returnClassToken: Bool = false,
                                      returnRegisterTokens: Bool = false,
                                      norm: Bool = true) -> [(patches: MLXArray, cls: MLXArray?, registers: MLXArray?)]
    {
        var (x, H, W) = prepareTokensWithMasks(x, masks: nil)

        let totalLayers = layer.count
        let layersToTake: [Int] = if let indices {
            indices
        } else {
            Array((totalLayers - n) ..< totalLayers)
        }

        var outputs: [MLXArray] = []

        for (i, block) in layer.enumerated() {
            let rope = ropeEmbed(H: H, W: W)
            let (blockOutput, _) = block(x, rope: rope, returnAttentionWeights: false)
            x = blockOutput

            if layersToTake.contains(i) {
                outputs.append(x)
            }
        }

        if norm {
            var normalizedOutputs: [MLXArray] = []
            for out in outputs {
                if untieCLSAndPatchNorms, let clsNorm {
                    let xNormClsReg = clsNorm(out[0..., ..<(numRegisterTokens + 1), 0...])
                    let xNormPatch = self.norm(out[0..., (numRegisterTokens + 1)..., 0...])
                    let normalized = concatenated([xNormClsReg, xNormPatch], axis: 1)
                    normalizedOutputs.append(normalized)
                } else {
                    normalizedOutputs.append(self.norm(out))
                }
            }
            outputs = normalizedOutputs
        }

        var results: [(patches: MLXArray, cls: MLXArray?, registers: MLXArray?)] = []

        for out in outputs {
            let clsTokens = out[0..., 0, 0...]
            let registerTokensOut = out[0..., 1 ..< (numRegisterTokens + 1), 0...]
            var patchTokens = out[0..., (numRegisterTokens + 1)..., 0...]

            if reshape {
                let B = patchTokens.dim(0)
                let D = patchTokens.dim(-1)
                patchTokens = patchTokens.reshaped(B, H, W, D).transposed(0, 3, 1, 2)
            }

            let cls = returnClassToken ? clsTokens : nil
            let registers = returnRegisterTokens ? registerTokensOut : nil

            results.append((patches: patchTokens, cls: cls, registers: registers))
        }

        return results
    }

    public func callAsFunction(_ x: MLXArray,
                               outputHiddenStates: Bool = false,
                               outputAttentions: Bool = false) -> DinoOutput
    {
        let (features, attentions) = forwardFeatures(x,
                                                     outputHiddenStates: outputHiddenStates,
                                                     outputAttentions: outputAttentions)

        let clsToken = features["x_norm_clstoken"]!
        let registerTokens = features["x_register_tokens"]!
        let patchTokens = features["x_norm_patchtokens"]!

        let lastHiddenState = concatenated([
            clsToken.expandedDimensions(axis: 1),
            registerTokens,
            patchTokens,
        ], axis: 1)

        let poolerOutput = clsToken

        let hiddenStatesOutput = outputHiddenStates ? features : nil

        return DinoOutput(lastHiddenState: lastHiddenState,
                          poolerOutput: poolerOutput,
                          hiddenStates: hiddenStatesOutput,
                          attentions: attentions)
    }
}
