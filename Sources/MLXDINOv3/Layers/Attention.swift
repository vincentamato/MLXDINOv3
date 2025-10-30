//
//  Attention.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

private func ropeRotateHalf(_ x: MLXArray) -> MLXArray {
    let D = x.dim(-1)
    let halfD = D / 2
    let x1 = x[.ellipsis, ..<halfD]
    let x2 = x[.ellipsis, halfD...]
    return concatenated([-x2, x1], axis: -1)
}

private func ropeApply(_ x: MLXArray, sin: MLXArray, cos: MLXArray) -> MLXArray {
    (x * cos) + (ropeRotateHalf(x) * sin)
}

public class SelfAttention: Module {
    let numHeads: Int
    let scale: Float
    let dim: Int
    @ModuleInfo var q_proj: Linear
    @ModuleInfo var k_proj: Linear
    @ModuleInfo var v_proj: Linear
    @ModuleInfo var o_proj: Linear

    public init(dim: Int,
                numHeads: Int = 8,
                queryBias: Bool = true,
                keyBias: Bool = false,
                valueBias: Bool = true,
                projBias: Bool = true)
    {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        scale = pow(Float(headDim), -0.5)
        self.dim = dim

        _q_proj.wrappedValue = Linear(dim, dim, bias: queryBias)
        _k_proj.wrappedValue = Linear(dim, dim, bias: keyBias)
        _v_proj.wrappedValue = Linear(dim, dim, bias: valueBias)
        _o_proj.wrappedValue = Linear(dim, dim, bias: projBias)
    }

    private func reshapeForAttention(_ x: MLXArray) -> MLXArray {
        let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
        let Dh = D / numHeads
        let reshaped = x.reshaped(B, N, numHeads, Dh)
        return reshaped.swappedAxes(1, 2)
    }

    private func applyRope(q: MLXArray,
                           k: MLXArray,
                           rope: (MLXArray, MLXArray)) -> (MLXArray, MLXArray)
    {
        let (cos, sin) = rope
        var qCast = q.asType(sin.dtype)
        var kCast = k.asType(sin.dtype)

        let N = q.dim(-2)
        let prefix = N - sin.dim(-2)
        precondition(prefix >= 0)

        if prefix > 0 {
            let qPrefix = qCast[.ellipsis, ..<prefix, 0...]
            let kPrefix = kCast[.ellipsis, ..<prefix, 0...]
            let qTail = qCast[.ellipsis, prefix..., 0...]
            let kTail = kCast[.ellipsis, prefix..., 0...]

            let qTailRot = ropeApply(qTail, sin: sin, cos: cos)
            let kTailRot = ropeApply(kTail, sin: sin, cos: cos)

            qCast = concatenated([qPrefix, qTailRot], axis: -2).asType(q.dtype)
            kCast = concatenated([kPrefix, kTailRot], axis: -2).asType(k.dtype)
        } else {
            qCast = ropeApply(qCast, sin: sin, cos: cos).asType(q.dtype)
            kCast = ropeApply(kCast, sin: sin, cos: cos).asType(k.dtype)
        }

        return (qCast, kCast)
    }

    private func computeAttention(q: MLXArray,
                                  k: MLXArray,
                                  v: MLXArray,
                                  rope: (MLXArray, MLXArray)? = nil,
                                  returnAttentionWeights: Bool = false) -> (output: MLXArray, attentionWeights: MLXArray?)
    {
        let (B, N, _) = (q.dim(0), q.dim(1), q.dim(2))
        let C = dim

        var qReshaped = reshapeForAttention(q)
        var kReshaped = reshapeForAttention(k)
        let vReshaped = reshapeForAttention(v)

        if let rope {
            (qReshaped, kReshaped) = applyRope(q: qReshaped, k: kReshaped, rope: rope)
        }

        let scores = matmul(qReshaped, kReshaped.swappedAxes(-1, -2)) * scale
        let attn = softmax(scores, axis: -1)

        var x = matmul(attn, vReshaped)
        x = x.swappedAxes(1, 2)
        x = x.reshaped(B, N, C)

        let attentionWeights = returnAttentionWeights ? attn : nil
        return (output: x, attentionWeights: attentionWeights)
    }

    public func callAsFunction(_ x: MLXArray,
                               rope: (MLXArray, MLXArray)? = nil,
                               returnAttentionWeights: Bool = false) -> (output: MLXArray, attentionWeights: MLXArray?)
    {
        let qOut = q_proj(x)
        let kOut = k_proj(x)
        let vOut = v_proj(x)

        let (attnV, attentionWeights) = computeAttention(q: qOut,
                                                         k: kOut,
                                                         v: vOut,
                                                         rope: rope,
                                                         returnAttentionWeights: returnAttentionWeights)

        let projOut = o_proj(attnV)

        return (output: projOut, attentionWeights: attentionWeights)
    }
}
