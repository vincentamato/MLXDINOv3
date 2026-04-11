import Foundation
import MLX
import MLXNN

public struct DinoOutput {
    public let lastHiddenState: MLXArray
    public let clsToken: MLXArray
    public let patchTokens: MLXArray
    public let hiddenStates: [MLXArray]?
}

public class DinoVisionTransformer: Module {
    let numRegisterTokens: Int
    @ModuleInfo var embeddings: Embeddings
    let ropeEmbed: RoPEPositionEmbedding
    @ModuleInfo(key: "layer") var layers: [TransformerBlock]
    @ModuleInfo var norm: LayerNorm

    public init(
        patchSize: Int,
        inChannels: Int,
        posEmbedRopeBase: Float,
        embedDim: Int,
        depth: Int,
        numHeads: Int,
        intermediateSize: Int,
        queryBias: Bool,
        keyBias: Bool,
        valueBias: Bool,
        layerscaleInit: Float,
        layerNormEps: Float,
        ffnBias: Bool,
        projBias: Bool,
        useGatedMlp: Bool,
        numRegisterTokens: Int
    ) {
        self.numRegisterTokens = numRegisterTokens

        _embeddings.wrappedValue = Embeddings(
            patchSize: patchSize,
            inChannels: inChannels,
            embedDim: embedDim,
            numRegisterTokens: numRegisterTokens)

        ropeEmbed = RoPEPositionEmbedding(
            embedDim: embedDim,
            numHeads: numHeads,
            base: posEmbedRopeBase)

        var layerList: [TransformerBlock] = []
        for _ in 0 ..< depth {
            layerList.append(
                TransformerBlock(
                    dim: embedDim,
                    numHeads: numHeads,
                    intermediateSize: intermediateSize,
                    queryBias: queryBias,
                    keyBias: keyBias,
                    valueBias: valueBias,
                    projBias: projBias,
                    ffnBias: ffnBias,
                    initValues: layerscaleInit,
                    layerNormEps: layerNormEps,
                    useGatedMlp: useGatedMlp))
        }
        _layers.wrappedValue = layerList

        _norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: layerNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> DinoOutput {
        var x = embeddings.patchEmbeddings(x)
        let (B, H, W, _) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        x = x.reshaped(B, H * W, x.dim(-1))

        let cls = embeddings.clsToken
        let register = embeddings.registerTokens
        let clsExpanded = broadcast(cls, to: [B, cls.dim(1), cls.dim(2)])
        let registerExpanded = broadcast(register, to: [B, register.dim(1), register.dim(2)])
        x = concatenated([clsExpanded, registerExpanded, x], axis: 1)

        var allHiddenStates: [MLXArray] = []
        if outputHiddenStates {
            allHiddenStates.append(x)
        }

        let rope = ropeEmbed(H: H, W: W, dtype: x.dtype)

        for block in layers {
            x = block(x, rope: rope)
            if outputHiddenStates {
                allHiddenStates.append(x)
            }
        }

        let lastHiddenState = norm(x)
        let prefixEnd = numRegisterTokens + 1

        return DinoOutput(
            lastHiddenState: lastHiddenState,
            clsToken: lastHiddenState[0..., 0, 0...],
            patchTokens: lastHiddenState[0..., prefixEnd..., 0...],
            hiddenStates: outputHiddenStates ? allHiddenStates : nil)
    }
}

private struct DinoConfig: Decodable {
    let patchSize: Int
    let numChannels: Int
    let ropeTheta: Float
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let intermediateSize: Int
    let queryBias: Bool
    let keyBias: Bool
    let valueBias: Bool
    let layerscaleValue: Float
    let layerNormEps: Float
    let mlpBias: Bool
    let projBias: Bool
    let useGatedMlp: Bool
    let numRegisterTokens: Int
}

extension DinoVisionTransformer {
    public static func loadPretrained(from modelPath: String) throws -> DinoVisionTransformer {
        let modelURL = URL(fileURLWithPath: modelPath)

        let configData = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(DinoConfig.self, from: configData)

        let model = DinoVisionTransformer(
            patchSize: config.patchSize,
            inChannels: config.numChannels,
            posEmbedRopeBase: config.ropeTheta,
            embedDim: config.hiddenSize,
            depth: config.numHiddenLayers,
            numHeads: config.numAttentionHeads,
            intermediateSize: config.intermediateSize,
            queryBias: config.queryBias,
            keyBias: config.keyBias,
            valueBias: config.valueBias,
            layerscaleInit: config.layerscaleValue,
            layerNormEps: config.layerNormEps,
            ffnBias: config.mlpBias,
            projBias: config.projBias,
            useGatedMlp: config.useGatedMlp,
            numRegisterTokens: config.numRegisterTokens)

        let weights = try loadArrays(url: modelURL.appendingPathComponent("model.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        return model
    }
}
