//
//  ModelLoading.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public func loadPretrained(modelPath: String) throws -> DinoVisionTransformer {
    let modelURL = URL(fileURLWithPath: modelPath)
    let configURL = modelURL.appendingPathComponent("config.json")
    let weightsURL = modelURL.appendingPathComponent("model.safetensors")

    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode([String: AnyCodable].self, from: configData)

    let model = DinoVisionTransformer(imgSize: config["image_size"]?.intValue ?? 224,
                                      patchSize: config["patch_size"]?.intValue ?? 16,
                                      inChannels: config["num_channels"]?.intValue ?? 3,
                                      posEmbedRopeBase: config["rope_theta"]?.floatValue ?? 100.0,
                                      posEmbedRopeNormalizeCoords: config["rope_normalize_coords"]?.stringValue ?? "separate",
                                      posEmbedRopeRescale: config["pos_embed_rescale"]?.floatValue,
                                      posEmbedRopeDtype: config["rope_dtype"]?.stringValue ?? "fp32",
                                      embedDim: config["hidden_size"]?.intValue ?? 768,
                                      depth: config["num_hidden_layers"]?.intValue ?? 12,
                                      numHeads: config["num_attention_heads"]?.intValue ?? 12,
                                      ffnRatio: config["mlp_ratio"]?.floatValue ?? 4.0,
                                      queryBias: config["query_bias"]?.boolValue ?? true,
                                      keyBias: config["key_bias"]?.boolValue ?? false,
                                      valueBias: config["value_bias"]?.boolValue ?? true,
                                      layerscaleInit: config["layerscale_value"]?.floatValue,
                                      layerNormEps: config["layer_norm_eps"]?.floatValue ?? 1e-05,
                                      ffnBias: config["mlp_bias"]?.boolValue ?? true,
                                      projBias: config["proj_bias"]?.boolValue ?? true,
                                      useGatedMLP: config["use_gated_mlp"]?.boolValue ?? false,
                                      numRegisterTokens: config["num_register_tokens"]?.intValue ?? 0,
                                      untieCLSAndPatchNorms: config["use_separate_norms_for_cls_and_patches"]?.boolValue ?? false)

    let weights = try loadArrays(url: weightsURL)
    try loadWeights(model: model, weights: weights)

    print("Loaded pretrained DinoVisionTransformer from \(modelPath)")

    return model
}

struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intValue = try? container.decode(Int.self) {
            value = intValue
        } else if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
        } else if let stringValue = try? container.decode(String.self) {
            value = stringValue
        } else if let boolValue = try? container.decode(Bool.self) {
            value = boolValue
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let intValue = value as? Int {
            try container.encode(intValue)
        } else if let doubleValue = value as? Double {
            try container.encode(doubleValue)
        } else if let stringValue = value as? String {
            try container.encode(stringValue)
        } else if let boolValue = value as? Bool {
            try container.encode(boolValue)
        }
    }

    var intValue: Int? { value as? Int }
    var floatValue: Float? {
        if let double = value as? Double {
            return Float(double)
        }
        return value as? Float
    }

    var stringValue: String? { value as? String }
    var boolValue: Bool? { value as? Bool }
}

private func loadArrays(url: URL) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: url)
}

private func loadWeights(model: DinoVisionTransformer, weights: [String: MLXArray]) throws {
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
    print("All weights loaded and verified")
}
