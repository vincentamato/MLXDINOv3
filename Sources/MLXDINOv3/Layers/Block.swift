//
//  Block.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public class SelfAttentionBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attention: SelfAttention
    @ModuleInfo var layer_scale1: LayerScale
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: Module
    let mlpForward: (MLXArray) -> MLXArray
    @ModuleInfo var layer_scale2: LayerScale

    public init(dim: Int,
                numHeads: Int,
                ffnRatio: Float = 4.0,
                queryBias: Bool = true,
                keyBias: Bool = false,
                valueBias: Bool = true,
                projBias: Bool = true,
                ffnBias: Bool = true,
                initValues: Float? = nil,
                layerNormEps: Float = 1e-05,
                useGatedMLP: Bool = false)
    {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: layerNormEps)

        _attention.wrappedValue = SelfAttention(dim: dim,
                                                numHeads: numHeads,
                                                queryBias: queryBias,
                                                keyBias: keyBias,
                                                valueBias: valueBias,
                                                projBias: projBias)

        _layer_scale1.wrappedValue = LayerScale(dim: dim, initValues: initValues ?? 1.0)

        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: layerNormEps)

        let mlpHiddenDim = Int(Float(dim) * ffnRatio)
        if useGatedMLP {
            let swiglu = SwiGLUFFN(inFeatures: dim,
                                   hiddenFeatures: mlpHiddenDim,
                                   bias: ffnBias)
            _mlp.wrappedValue = swiglu
            mlpForward = { x in swiglu(x) }
        } else {
            let mlpLayer = MLP(inFeatures: dim,
                               hiddenFeatures: mlpHiddenDim,
                               bias: ffnBias)
            _mlp.wrappedValue = mlpLayer
            mlpForward = { x in mlpLayer(x) }
        }

        _layer_scale2.wrappedValue = LayerScale(dim: dim, initValues: initValues ?? 1.0)
    }

    public func callAsFunction(_ x: MLXArray,
                               rope: (MLXArray, MLXArray)? = nil,
                               returnAttentionWeights: Bool = false) -> (output: MLXArray, attentionWeights: MLXArray?)
    {
        let norm1Out = norm1(x)
        let (attnOut, attentionWeights) = attention(norm1Out,
                                                    rope: rope,
                                                    returnAttentionWeights: returnAttentionWeights)
        let ls1Out = layer_scale1(attnOut)
        let xAttn = x + ls1Out

        let norm2Out = norm2(xAttn)
        let mlpOut = mlpForward(norm2Out)
        let ls2Out = layer_scale2(mlpOut)
        let xFFN = xAttn + ls2Out

        return (output: xFFN, attentionWeights: attentionWeights)
    }
}
