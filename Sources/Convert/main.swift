//
//  main.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import Hub
import MLX
import MLXFast
import MLXNN

func printUsage() {
    print("""
    USAGE: Convert <model-id> <output-dir>

    ARGUMENTS:
      <model-id>    Hugging Face model ID (e.g., 'facebook/dinov3-vith16plus-pretrain-lvd1689m')
      <output-dir>  Output directory for converted model

    EXAMPLES:
      Convert facebook/dinov3-vits16plus-pretrain-lvd1689m ./output
      Convert facebook/dinov3-vitb16-pretrain-lvd1689m ./output
    """)
}

func convert(modelId: String, outputDir: String) async throws {
    let log = { (msg: String) in print(msg) }
    let logWarning = { (msg: String) in print("WARNING: \(msg)") }

    log("Converting \(modelId) to MLX format")

    log("Downloading model from Hugging Face: \(modelId)")

    let hub = HubApi()

    log("Downloading config.json...")
    let configURL = try await hub.snapshot(from: modelId, matching: ["config.json"])
    let configPath = configURL.appendingPathComponent("config.json")

    let configData = try Data(contentsOf: configPath)
    guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw ConversionError.invalidConfig
    }

    log("Downloading model.safetensors...")
    let weightsURL = try await hub.snapshot(from: modelId, matching: ["model.safetensors"])
    let weightsPath = weightsURL.appendingPathComponent("model.safetensors")

    log("Loading PyTorch weights...")
    let safeTensorsUrl = URL(fileURLWithPath: weightsPath.path)
    let ptWeights = try MLX.loadArrays(url: safeTensorsUrl)

    log("Loaded \(ptWeights.count) weight tensors from PyTorch model")

    let modelType = config["model_type"] as? String ?? "dinov3_vit"
    log("Model type: \(modelType)")

    log("Converting weights...")
    let mlxWeights: [String: MLXArray]
    if modelType.lowercased().contains("convnext") {
        throw ConversionError.unsupportedModelType(modelType: modelType)
    }

    mlxWeights = mapViTWeights(ptWeights: ptWeights, logWarning: logWarning)

    log("Converted \(mlxWeights.count) weight tensors to MLX format")

    try saveMLXModel(mlxWeights: mlxWeights, config: config, outputDir: outputDir, log: log)

    log("Conversion complete!")
}

func mapViTWeights(ptWeights: [String: MLXArray], logWarning _: (String) -> Void) -> [String: MLXArray] {
    var mlxWeights: [String: MLXArray] = [:]

    for (ptKey, ptValue) in ptWeights {
        let mlxKey = ptKey

        if ptKey == "embeddings.patch_embeddings.weight" {
            let transposed = ptValue.transposed(0, 2, 3, 1)
            mlxWeights[ptKey] = transposed
            continue
        }

        mlxWeights[mlxKey] = ptValue
    }

    return mlxWeights
}

func saveMLXModel(mlxWeights: [String: MLXArray],
                  config: [String: Any],
                  outputDir: String,
                  log: (String) -> Void) throws
{
    let outputURL = URL(fileURLWithPath: outputDir)
    try FileManager.default.createDirectory(at: outputURL,
                                            withIntermediateDirectories: true)

    let configPath = outputURL.appendingPathComponent("config.json")
    log("Saving config to \(configPath.path)")

    let configData = try JSONSerialization.data(withJSONObject: config,
                                                options: [.prettyPrinted, .sortedKeys])
    try configData.write(to: configPath)

    let weightsPath = outputURL.appendingPathComponent("model.safetensors")
    log("Saving weights to \(weightsPath.path)")

    try save(arrays: mlxWeights, url: weightsPath)

    log("Model saved successfully to \(outputDir)")
}

enum ConversionError: Error, LocalizedError {
    case unsupportedModelType(modelType: String)
    case invalidConfig
    case downloadFailed(String)
    case weightConversionFailed(String)

    var errorDescription: String? {
        switch self {
        case let .unsupportedModelType(modelType):
            "Unsupported model type: \(modelType)"
        case .invalidConfig:
            "Invalid or missing config.json"
        case let .downloadFailed(message):
            "Download failed: \(message)"
        case let .weightConversionFailed(message):
            "Weight conversion failed: \(message)"
        }
    }
}

let args = CommandLine.arguments

guard args.count >= 3 else {
    printUsage()
    exit(1)
}

let modelId = args[1]
let outputDir = args[2]

Task {
    do {
        try await convert(modelId: modelId, outputDir: outputDir)
        exit(0)
    } catch {
        print("Error: \(error)")
        exit(1)
    }
}

RunLoop.main.run()
