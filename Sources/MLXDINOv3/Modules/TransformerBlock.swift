import MLX
import MLXNN

final class TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attention: Attention
    @ModuleInfo(key: "layer_scale1") var layerScale1: LayerScale
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "layer_scale2") var layerScale2: LayerScale

    init(
        dim: Int,
        numHeads: Int,
        intermediateSize: Int,
        queryBias: Bool,
        keyBias: Bool,
        valueBias: Bool,
        projBias: Bool,
        ffnBias: Bool,
        initValues: Float,
        layerNormEps: Float,
        useGatedMlp: Bool
    ) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: layerNormEps)

        _attention.wrappedValue = Attention(
            dim: dim,
            numHeads: numHeads,
            queryBias: queryBias,
            keyBias: keyBias,
            valueBias: valueBias,
            projBias: projBias)

        _layerScale1.wrappedValue = LayerScale(dim: dim, initValues: initValues)

        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: layerNormEps)

        _mlp.wrappedValue =
            useGatedMlp
            ? SwiGLUFFN(inFeatures: dim, hiddenFeatures: intermediateSize, bias: ffnBias)
            : MLP(inFeatures: dim, hiddenFeatures: intermediateSize, bias: ffnBias)

        _layerScale2.wrappedValue = LayerScale(dim: dim, initValues: initValues)
    }

    func callAsFunction(
        _ x: MLXArray,
        rope: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let xAttn = x + layerScale1(attention(norm1(x), rope: rope))
        return xAttn + layerScale2(mlp(norm2(xAttn)))
    }
}
