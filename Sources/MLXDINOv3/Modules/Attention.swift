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

final class Attention: Module {
    let numHeads: Int
    let scale: Float
    let dim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(
        dim: Int,
        numHeads: Int,
        queryBias: Bool,
        keyBias: Bool,
        valueBias: Bool,
        projBias: Bool
    ) {
        self.numHeads = numHeads
        let headDim = dim / numHeads
        scale = pow(Float(headDim), -0.5)
        self.dim = dim

        _qProj.wrappedValue = Linear(dim, dim, bias: queryBias)
        _kProj.wrappedValue = Linear(dim, dim, bias: keyBias)
        _vProj.wrappedValue = Linear(dim, dim, bias: valueBias)
        _oProj.wrappedValue = Linear(dim, dim, bias: projBias)
    }

    private func reshapeForAttention(_ x: MLXArray) -> MLXArray {
        let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
        let Dh = D / numHeads
        return x.reshaped(B, N, numHeads, Dh).swappedAxes(1, 2)
    }

    private func applyRope(
        q: MLXArray,
        k: MLXArray,
        rope: (MLXArray, MLXArray)
    ) -> (MLXArray, MLXArray) {
        let (cos, sin) = rope

        // Only patch tokens get RoPE
        let prefix = q.dim(-2) - sin.dim(-2)
        precondition(prefix >= 0)

        if prefix > 0 {
            let qPrefix = q[.ellipsis, ..<prefix, 0...]
            let kPrefix = k[.ellipsis, ..<prefix, 0...]
            let qTail = ropeApply(q[.ellipsis, prefix..., 0...], sin: sin, cos: cos)
            let kTail = ropeApply(k[.ellipsis, prefix..., 0...], sin: sin, cos: cos)
            return (
                concatenated([qPrefix, qTail], axis: -2),
                concatenated([kPrefix, kTail], axis: -2)
            )
        } else {
            return (
                ropeApply(q, sin: sin, cos: cos),
                ropeApply(k, sin: sin, cos: cos)
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        rope: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let (B, N) = (x.dim(0), x.dim(1))

        var q = reshapeForAttention(qProj(x))
        var k = reshapeForAttention(kProj(x))
        let v = reshapeForAttention(vProj(x))

        if let rope {
            (q, k) = applyRope(q: q, k: k, rope: rope)
        }

        var attnOut = scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: nil)
        attnOut = attnOut.swappedAxes(1, 2).reshaped(B, N, dim)

        return oProj(attnOut)
    }
}
