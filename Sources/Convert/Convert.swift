import Foundation
import Hub
import MLX

@main
struct Convert {
    static func main() async {
        let args = CommandLine.arguments

        guard args.count >= 3 else {
            printUsage()
            exit(1)
        }

        let modelId = args[1]
        let outputDir = args[2]

        do {
            try await convert(modelId: modelId, outputDir: outputDir)
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }
}

func printUsage() {
    print(
        """
        USAGE: ./mlx-run.sh convert <model-id> <output-dir>

        ARGUMENTS:
          <model-id>    Hugging Face model ID (e.g., 'facebook/dinov3-vith16plus-pretrain-lvd1689m')
          <output-dir>  Output directory for converted model

        EXAMPLES:
          ./mlx-run.sh convert facebook/dinov3-vits16plus-pretrain-lvd1689m ./output
          ./mlx-run.sh convert facebook/dinov3-vitb16-pretrain-lvd1689m ./output
        """)
}

func convert(modelId: String, outputDir: String) async throws {
    print("Converting \(modelId) to MLX format")

    let hub = HubApi()

    print("Downloading model from Hugging Face...")
    let repoURL = try await hub.snapshot(
        from: modelId, matching: ["config.json", "model.safetensors"])

    let configData = try Data(contentsOf: repoURL.appendingPathComponent("config.json"))
    guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw ConversionError.invalidConfig
    }

    let weightsPath = repoURL.appendingPathComponent("model.safetensors")

    print("Loading PyTorch weights...")
    let safeTensorsUrl = URL(fileURLWithPath: weightsPath.path)
    let ptWeights = try loadArrays(url: safeTensorsUrl)

    print("Loaded \(ptWeights.count) weight tensors from PyTorch model")

    let modelType = config["model_type"] as? String ?? "dinov3_vit"
    print("Model type: \(modelType)")

    print("Converting weights...")
    if modelType.lowercased().contains("convnext") {
        throw ConversionError.unsupportedModelType(modelType: modelType)
    }

    let mlxWeights = mapViTWeights(ptWeights: ptWeights)

    print("Converted \(mlxWeights.count) weight tensors to MLX format")

    try saveMLXModel(mlxWeights: mlxWeights, config: config, outputDir: outputDir)

    print("Conversion complete!")
}

func mapViTWeights(ptWeights: [String: MLXArray]) -> [String: MLXArray] {
    var mlxWeights: [String: MLXArray] = [:]

    for (ptKey, ptValue) in ptWeights {
        if ptKey == "embeddings.mask_token" {
            continue
        }

        if ptKey == "embeddings.patch_embeddings.weight" {
            mlxWeights[ptKey] = ptValue.transposed(0, 2, 3, 1)
            continue
        }

        mlxWeights[ptKey] = ptValue
    }

    return mlxWeights
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

enum ConversionError: Error, CustomStringConvertible {
    case unsupportedModelType(modelType: String)
    case invalidConfig

    var description: String {
        switch self {
        case .unsupportedModelType(let modelType):
            "Unsupported model type: \(modelType)"
        case .invalidConfig:
            "Invalid or missing config.json"
        }
    }
}
