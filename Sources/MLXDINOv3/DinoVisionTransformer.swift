import Foundation
import MLX
import MLXNN

/// The result of a ``DinoVisionTransformer`` forward pass.
public struct DinoOutput {
    /// All tokens after the final layer norm: `[batch, 1 + registers + patches, embedDim]`.
    /// The order is the CLS token, then the register tokens, then the patch tokens.
    public let lastHiddenState: MLXArray

    /// The CLS token of ``lastHiddenState``: `[batch, embedDim]`.
    public let clsToken: MLXArray

    /// The patch tokens of ``lastHiddenState`` in row-major order: `[batch, patches, embedDim]`.
    public let patchTokens: MLXArray

    /// Per-layer tokens, present when `outputHiddenStates` was `true`.
    ///
    /// `hiddenStates[0]` is the embedding output and `hiddenStates[i]` is the output of block `i`,
    /// so there are `depth + 1` entries. None of them has the final layer norm applied, including
    /// the last one: use ``lastHiddenState`` for the normalized output. (Hugging Face's
    /// `output_hidden_states` returns the normalized output as its final entry instead.)
    public let hiddenStates: [MLXArray]?
}

/// DINOv3 vision transformer.
///
/// Load converted weights with ``loadPretrained(from:)`` and call the model with a channels-last
/// image batch, for example the output of ``ImageProcessor``:
///
/// ```swift
/// let model = try DinoVisionTransformer.loadPretrained(from: "Models/dinov3-vits16-mlx")
/// let outputs = model(try ImageProcessor()(image))
/// ```
///
/// The model is not thread-safe; call it from one task at a time.
public class DinoVisionTransformer: Module {
    /// Activation used inside the feed-forward layers (`hidden_act` in `config.json`).
    public enum Activation: String, Sendable, CaseIterable {
        case gelu
        case silu
        case relu

        /// Maps a Hugging Face `hidden_act` name to an activation, or `nil` if it is unsupported.
        public init?(configName: String) {
            switch configName.lowercased() {
            case "gelu": self = .gelu
            case "silu", "swish": self = .silu
            case "relu": self = .relu
            default: return nil
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            switch self {
            case .gelu: MLXNN.gelu(x)
            case .silu: MLXNN.silu(x)
            case .relu: MLXNN.relu(x)
            }
        }
    }

    /// Side length of the square patches, in pixels.
    public let patchSize: Int
    /// Number of input channels.
    public let inChannels: Int
    /// Width of every token (`hidden_size`).
    public let embedDim: Int
    /// Number of transformer blocks.
    public let depth: Int
    /// Number of attention heads per block.
    public let numHeads: Int
    /// Number of register tokens between the CLS token and the patch tokens.
    public let numRegisterTokens: Int

    @ModuleInfo var embeddings: Embeddings
    let ropeEmbed: RoPEPositionEmbedding
    @ModuleInfo(key: "layer") var layers: [TransformerBlock]
    @ModuleInfo var norm: LayerNorm

    /// Creates a model with freshly initialised weights.
    ///
    /// Most callers want ``loadPretrained(from:)``, which reads these values from `config.json`.
    ///
    /// - Parameters:
    ///   - embedDim: token width; must be a multiple of `4 * numHeads`.
    ///   - intermediateSize: hidden width of the feed-forward layers.
    ///   - layerscaleInit: initial value of the layer-scale weights.
    ///   - useGatedMlp: use the gated (SwiGLU-style) feed-forward layer.
    ///   - activation: feed-forward activation. `nil` picks the pairing of the released
    ///     checkpoints: SiLU for the gated layer, GELU otherwise.
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
        numRegisterTokens: Int,
        activation: Activation? = nil
    ) {
        precondition(patchSize > 0 && inChannels > 0, "patchSize and inChannels must be positive")
        precondition(
            depth >= 0 && numRegisterTokens >= 0, "depth and numRegisterTokens must be >= 0")

        self.patchSize = patchSize
        self.inChannels = inChannels
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
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
                    useGatedMlp: useGatedMlp,
                    activation: activation))
        }
        _layers.wrappedValue = layerList

        _norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: layerNormEps)
    }

    /// Runs the model.
    ///
    /// - Parameters:
    ///   - x: a channels-last image batch `[batch, height, width, inChannels]`, already
    ///     normalized. Height and width need not match and need not be multiples of
    ///     ``patchSize``; a partial patch at the right or bottom edge is dropped. A malformed
    ///     shape is a programmer error and stops with a message naming the expected layout.
    ///   - outputHiddenStates: also return the tokens after every block.
    ///
    /// The input is cast to the precision of the weights, as the reference implementation does,
    /// so a half-precision checkpoint runs (and returns) half precision. To run such a checkpoint
    /// in single precision, cast the weights once: `model.apply { $0.asType(.float32) }`.
    public func callAsFunction(
        _ x: MLXArray,
        outputHiddenStates: Bool = false
    ) -> DinoOutput {
        precondition(
            x.ndim == 4 && x.dim(3) == inChannels,
            "DinoVisionTransformer expects a channels-last batch "
                + "[batch, height, width, \(inChannels)] but got \(x.shape)"
                + (x.ndim == 4 && x.dim(1) == inChannels
                    ? ". This looks channels-first; use x.transposed(0, 2, 3, 1)." : "."))
        precondition(
            x.dim(1) >= patchSize && x.dim(2) >= patchSize,
            "Input \(x.dim(1))x\(x.dim(2)) is smaller than one \(patchSize)x\(patchSize) patch.")

        let weightDType = embeddings.patchEmbeddings.weight.dtype
        var x = embeddings.patchEmbeddings(x.asType(weightDType))
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
    let modelType: String?
    let hiddenAct: String?
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

    /// Checks everything that would otherwise trap while building the model.
    func validated() throws -> DinoVisionTransformer.Activation? {
        typealias LoadError = DinoVisionTransformer.LoadError

        if let modelType, modelType != "dinov3_vit" {
            throw LoadError.invalidConfig(
                "model_type is \"\(modelType)\"; only \"dinov3_vit\" is supported")
        }
        let positive = [
            ("patch_size", patchSize), ("num_channels", numChannels), ("hidden_size", hiddenSize),
            ("num_attention_heads", numAttentionHeads), ("intermediate_size", intermediateSize),
        ]
        for (name, value) in positive where value <= 0 {
            throw LoadError.invalidConfig("\(name) must be positive, got \(value)")
        }
        for (name, value) in [
            ("num_hidden_layers", numHiddenLayers), ("num_register_tokens", numRegisterTokens),
        ] where value < 0 {
            throw LoadError.invalidConfig("\(name) must not be negative, got \(value)")
        }
        guard hiddenSize % (4 * numAttentionHeads) == 0 else {
            throw LoadError.invalidConfig(
                "hidden_size (\(hiddenSize)) must be a multiple of 4 * num_attention_heads "
                    + "(\(numAttentionHeads))")
        }
        guard let hiddenAct else { return nil }
        guard let activation = DinoVisionTransformer.Activation(configName: hiddenAct) else {
            let supported = DinoVisionTransformer.Activation.allCases.map(\.rawValue)
            throw LoadError.invalidConfig(
                "hidden_act \"\(hiddenAct)\" is not supported (supported: "
                    + supported.joined(separator: ", ") + ")")
        }
        return activation
    }
}

extension DinoVisionTransformer {
    /// Why ``loadPretrained(from:)`` rejected a model directory.
    public enum LoadError: Error, Equatable, CustomStringConvertible, LocalizedError {
        /// `config.json` is missing a value, or has one the model cannot be built from.
        case invalidConfig(String)
        /// The patch-embedding weights are still in PyTorch layout.
        case weightsNotConverted(key: String, shape: [Int])
        /// The weights file lacks tensors the model needs.
        case missingWeights([String])
        /// The weights file has tensors the model has no place for.
        case unexpectedWeights([String])
        /// A tensor has a different shape from the one `config.json` implies.
        case shapeMismatch(key: String, expected: [Int], actual: [Int])

        public var description: String {
            switch self {
            case .invalidConfig(let reason):
                return "Invalid config.json: \(reason)."
            case .weightsNotConverted(let key, let shape):
                return "\(key) has shape \(shape), which is the PyTorch layout. "
                    + "Convert the checkpoint with the Convert tool before loading it."
            case .missingWeights(let keys):
                return "model.safetensors is missing \(keys.count) tensor(s) that config.json "
                    + "calls for: \(Self.preview(keys)). The file and config.json do not match."
            case .unexpectedWeights(let keys):
                return "model.safetensors has \(keys.count) tensor(s) the model does not use: "
                    + "\(Self.preview(keys)). The file and config.json do not match."
            case .shapeMismatch(let key, let expected, let actual):
                return "\(key) has shape \(actual) but config.json implies \(expected)."
            }
        }

        public var errorDescription: String? { description }

        private static func preview(_ keys: [String]) -> String {
            let shown = keys.prefix(5).joined(separator: ", ")
            return keys.count > 5 ? shown + ", …" : shown
        }
    }

    /// Loads a model from a directory holding `config.json` and `model.safetensors`, as written
    /// by the `Convert` tool.
    ///
    /// The weights are checked against the model that `config.json` describes: every tensor must
    /// be present, used and of the expected shape. A mismatch throws a ``LoadError`` rather than
    /// leaving part of the model randomly initialised.
    ///
    /// - Parameter modelPath: path of the model directory.
    /// - Throws: ``LoadError``, or a file error if either file cannot be read.
    public static func loadPretrained(from modelPath: String) throws -> DinoVisionTransformer {
        let modelURL = URL(fileURLWithPath: modelPath)

        let configData = try Data(contentsOf: modelURL.appendingPathComponent("config.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config: DinoConfig
        do {
            config = try decoder.decode(DinoConfig.self, from: configData)
        } catch let error as DecodingError {
            throw LoadError.invalidConfig(describe(error))
        }
        let activation = try config.validated()

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
            numRegisterTokens: config.numRegisterTokens,
            activation: activation)

        var weights = try loadArrays(url: modelURL.appendingPathComponent("model.safetensors"))
        // The mask token is only used in pre-training. Checkpoints converted by 1.x kept it.
        weights["embeddings.mask_token"] = nil
        // 2.0.0 exposed the fixed rotary frequencies as a parameter, so a checkpoint saved from
        // one of its models carries them.
        weights["ropeEmbed.invFreq"] = nil

        try model.check(weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

        return model
    }

    /// Compares a flat weights dictionary with this model's parameters.
    ///
    /// `update(parameters:verify:)` alone is not enough: it ignores surplus entries of a module
    /// array, so a file with more layers than `config.json` would load as a truncated model.
    private func check(_ weights: [String: MLXArray]) throws {
        let expected = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })

        let patchKey = "embeddings.patch_embeddings.weight"
        if let actual = weights[patchKey]?.shape, let wanted = expected[patchKey],
            actual != wanted, actual == [wanted[0], wanted[3], wanted[1], wanted[2]]
        {
            throw LoadError.weightsNotConverted(key: patchKey, shape: actual)
        }

        let unexpected = weights.keys.filter { expected[$0] == nil }.sorted()
        guard unexpected.isEmpty else { throw LoadError.unexpectedWeights(unexpected) }

        let missing = expected.keys.filter { weights[$0] == nil }.sorted()
        guard missing.isEmpty else { throw LoadError.missingWeights(missing) }

        for key in expected.keys.sorted() {
            if let actual = weights[key]?.shape, let wanted = expected[key], actual != wanted {
                throw LoadError.shapeMismatch(key: key, expected: wanted, actual: actual)
            }
        }
    }

    private static func describe(_ error: DecodingError) -> String {
        func name(_ path: [any CodingKey], _ key: (any CodingKey)? = nil) -> String {
            let parts = (path + (key.map { [$0] } ?? [])).map { snakeCase($0.stringValue) }
            return "\"" + parts.joined(separator: ".") + "\""
        }
        switch error {
        case .keyNotFound(let key, let context):
            return "\(name(context.codingPath, key)) is missing"
        case .valueNotFound(_, let context):
            return "\(name(context.codingPath)) is null"
        case .typeMismatch(let type, let context):
            return "\(name(context.codingPath)) is not a valid \(type)"
        case .dataCorrupted(let context):
            return context.codingPath.isEmpty
                ? "the file is not valid JSON" : "\(name(context.codingPath)) is malformed"
        @unknown default:
            return String(describing: error)
        }
    }

    private static func snakeCase(_ camel: String) -> String {
        camel.reduce(into: "") { result, character in
            if character.isUppercase {
                result += "_" + character.lowercased()
            } else {
                result.append(character)
            }
        }
    }
}
