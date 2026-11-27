import Foundation
import Hub
import MLX

@main
struct Convert {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())

        var stripPrefix: String?
        if let flag = args.firstIndex(of: "--strip-prefix") {
            guard flag + 1 < args.count else {
                printUsage()
                exit(1)
            }
            stripPrefix = args[flag + 1]
            args.removeSubrange(flag ... flag + 1)
        }

        guard args.count == 2 else {
            printUsage()
            exit(1)
        }

        do {
            try await convert(source: args[0], outputDir: args[1], stripPrefix: stripPrefix)
        } catch Hub.HubClientError.authorizationRequired {
            fputs(
                """
                Error: Hugging Face refused the download of \(args[0]).
                Meta's DINOv3 checkpoints are gated. To convert one:
                  1. Accept the license on https://huggingface.co/\(args[0])
                  2. Create an access token at https://huggingface.co/settings/tokens
                  3. Run this command again with HF_TOKEN=<your token> in the environment

                """, stderr)
            exit(1)
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }
}

func printUsage() {
    print(
        """
        USAGE: ./mlx-run.sh Convert <model> <output-dir> [--strip-prefix <prefix>]

        ARGUMENTS:
          <model>       Hugging Face model ID (e.g. 'facebook/dinov3-vith16plus-pretrain-lvd1689m')
                        or a local directory holding config.json and model.safetensors
          <output-dir>  Output directory for the converted model

        OPTIONS:
          --strip-prefix <prefix>
                        Keep only the tensors whose name starts with <prefix> and remove the
                        prefix. Use it to extract the backbone of a fine-tuned checkpoint,
                        e.g. --strip-prefix backbone.

        EXAMPLES:
          ./mlx-run.sh Convert facebook/dinov3-vits16plus-pretrain-lvd1689m ./output
          ./mlx-run.sh Convert ./my-finetuned-checkpoint ./output --strip-prefix backbone.

        Meta's checkpoints are gated: accept the license on the model page and set HF_TOKEN.
        """)
}

func convert(source: String, outputDir: String, stripPrefix: String? = nil) async throws {
    print("Converting \(source) to MLX format")

    var isDirectory: ObjCBool = false
    let isLocal =
        FileManager.default.fileExists(atPath: source, isDirectory: &isDirectory)
        && isDirectory.boolValue

    let sourceURL: URL
    let config: [String: Any]
    if isLocal {
        sourceURL = URL(fileURLWithPath: source)
        config = try loadConfig(from: sourceURL)
    } else {
        // Check the config before fetching the weights, which can be many gigabytes.
        let hub = HubApi()
        print("Downloading config.json from Hugging Face...")
        let repoURL = try await hub.snapshot(from: source, matching: ["config.json"])
        config = try loadConfig(from: repoURL)
        print("Downloading model.safetensors from Hugging Face...")
        sourceURL = try await hub.snapshot(from: source, matching: ["model.safetensors"])
    }

    let weightsURL = sourceURL.appendingPathComponent("model.safetensors")
    guard FileManager.default.fileExists(atPath: weightsURL.path) else {
        throw ConversionError.missingFile("model.safetensors", in: sourceURL.path)
    }

    print("Loading PyTorch weights...")
    let ptWeights = try loadArrays(url: weightsURL)
    print("Loaded \(ptWeights.count) weight tensors")

    let (mlxWeights, notes) = try convertWeights(
        ptWeights, config: config, stripPrefix: stripPrefix)
    for note in notes {
        print("Note: \(note)")
    }
    print("Converted \(mlxWeights.count) weight tensors to MLX format")

    try saveMLXModel(mlxWeights: mlxWeights, config: config, outputDir: outputDir)

    print("Conversion complete!")
}

func loadConfig(from directory: URL) throws -> [String: Any] {
    let url = directory.appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw ConversionError.missingFile("config.json", in: directory.path)
    }
    guard
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            as? [String: Any]
    else {
        throw ConversionError.invalidConfig
    }
    let modelType = config["model_type"] as? String ?? "dinov3_vit"
    guard modelType == "dinov3_vit" else {
        throw ConversionError.unsupportedModelType(modelType: modelType)
    }
    return config
}

let patchWeightKey = "embeddings.patch_embeddings.weight"
let maskTokenKey = "embeddings.mask_token"

/// Converts a PyTorch DINOv3 ViT state dict to the layout `DinoVisionTransformer` loads.
///
/// Tensor names may carry a prefix such as `backbone.`; the patch-embedding weight and the mask
/// token are recognised either way. Returns the converted tensors and human-readable notes.
func convertWeights(
    _ ptWeights: [String: MLXArray], config: [String: Any], stripPrefix: String? = nil
) throws -> (weights: [String: MLXArray], notes: [String]) {
    func matches(_ key: String, _ name: String) -> Bool {
        key == name || key.hasSuffix("." + name)
    }

    var notes: [String] = []
    var weights = ptWeights

    if let stripPrefix, !stripPrefix.isEmpty {
        let kept = weights.filter { $0.key.hasPrefix(stripPrefix) }
        guard !kept.isEmpty else { throw ConversionError.prefixNotFound(stripPrefix) }
        let dropped = weights.count - kept.count
        if dropped > 0 {
            notes.append("dropped \(dropped) tensor(s) without the prefix \"\(stripPrefix)\"")
        }
        weights = Dictionary(
            uniqueKeysWithValues: kept.map {
                (String($0.key.dropFirst(stripPrefix.count)), $0.value)
            })
    }

    guard let patchKey = weights.keys.first(where: { matches($0, patchWeightKey) }) else {
        throw ConversionError.missingTensor(patchWeightKey)
    }

    var mlxWeights: [String: MLXArray] = [:]
    for (key, value) in weights {
        if matches(key, maskTokenKey) {
            // Only used in pre-training.
            continue
        }
        if key == patchKey {
            mlxWeights[key] = try convertPatchWeight(value, key: key, config: config, notes: &notes)
            continue
        }
        mlxWeights[key] = value
    }

    if patchKey != patchWeightKey {
        let prefix = String(patchKey.dropLast(patchWeightKey.count))
        notes.append(
            "tensor names start with \"\(prefix)\". DinoVisionTransformer.loadPretrained expects "
                + "names without a prefix; pass --strip-prefix \(prefix) to extract the backbone")
    }

    return (mlxWeights, notes)
}

/// PyTorch stores the convolution as `[out, in, height, width]`, MLX as `[out, height, width, in]`.
func convertPatchWeight(
    _ value: MLXArray, key: String, config: [String: Any], notes: inout [String]
) throws -> MLXArray {
    guard value.ndim == 4 else {
        throw ConversionError.unexpectedShape(key: key, shape: value.shape)
    }
    if let hidden = config["hidden_size"] as? Int, let patch = config["patch_size"] as? Int {
        let channels = config["num_channels"] as? Int ?? 3
        let pytorch = [hidden, channels, patch, patch]
        let mlx = [hidden, patch, patch, channels]
        if value.shape == mlx, mlx != pytorch {
            notes.append("\(key) is already in MLX layout; left as it is")
            return value
        }
        guard value.shape == pytorch else {
            throw ConversionError.unexpectedShape(key: key, shape: value.shape)
        }
    }
    return value.transposed(0, 2, 3, 1)
}

func saveMLXModel(
    mlxWeights: [String: MLXArray],
    config: [String: Any],
    outputDir: String
) throws {
    let outputURL = URL(fileURLWithPath: outputDir)
    try FileManager.default.createDirectory(
        at: outputURL,
        withIntermediateDirectories: true)

    let configPath = outputURL.appendingPathComponent("config.json")
    print("Saving config to \(configPath.path)")

    let configData = try JSONSerialization.data(
        withJSONObject: config,
        options: [.prettyPrinted, .sortedKeys])
    try configData.write(to: configPath)

    let weightsPath = outputURL.appendingPathComponent("model.safetensors")
    print("Saving weights to \(weightsPath.path)")

    try save(arrays: mlxWeights, url: weightsPath)

    print("Model saved successfully to \(outputDir)")
}

enum ConversionError: Error, Equatable, CustomStringConvertible {
    case unsupportedModelType(modelType: String)
    case invalidConfig
    case missingFile(String, in: String)
    case missingTensor(String)
    case unexpectedShape(key: String, shape: [Int])
    case prefixNotFound(String)

    var description: String {
        switch self {
        case .unsupportedModelType(let modelType):
            "Unsupported model type \"\(modelType)\"; only DINOv3 ViT models (dinov3_vit) are supported"
        case .invalidConfig:
            "config.json is not a JSON object"
        case .missingFile(let name, let directory):
            "\(name) not found in \(directory)"
        case .missingTensor(let name):
            "No tensor named \(name) in the checkpoint; is this a DINOv3 ViT model?"
        case .unexpectedShape(let key, let shape):
            "\(key) has shape \(shape), which does not match config.json"
        case .prefixNotFound(let prefix):
            "No tensor name starts with \"\(prefix)\""
        }
    }
}
