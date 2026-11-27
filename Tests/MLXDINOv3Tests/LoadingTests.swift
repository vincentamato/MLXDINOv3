import Foundation
import MLX
import Testing

@testable import MLXDINOv3

extension AllTests {
    /// `loadPretrained` must reject files that do not match, not load part of a model.
    @Suite("Loading")
    struct LoadingTests {
        typealias LoadError = DinoVisionTransformer.LoadError

        let directory = Fixtures.model("plain")

        func weights() throws -> [String: MLXArray] {
            try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        }

        func config(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            var json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            edit(&json)
            return try JSONSerialization.data(withJSONObject: json)
        }

        func load(weights: [String: MLXArray], config: Data) throws -> DinoVisionTransformer {
            let scratch = try TemporaryDirectory()
            try scratch.writeModel(weights: weights, config: config)
            return try DinoVisionTransformer.loadPretrained(from: scratch.url.path)
        }

        /// Runs `load` and returns the `LoadError` it must throw.
        func loadError(weights: [String: MLXArray], config: Data) throws -> LoadError {
            do {
                _ = try load(weights: weights, config: config)
            } catch let error as LoadError {
                return error
            }
            Issue.record("loadPretrained accepted a model it should have rejected")
            throw CancellationError()
        }

        @Test("A PyTorch-layout checkpoint is reported as not converted")
        func unconvertedCheckpoint() throws {
            var raw = try loadArrays(
                url: directory.appendingPathComponent("pytorch_model.safetensors"))
            #expect(
                try loadError(weights: raw, config: config())
                    == .weightsNotConverted(
                        key: "embeddings.patch_embeddings.weight", shape: [32, 3, 16, 16]))
            raw["embeddings.mask_token"] = nil
            #expect(
                try loadError(weights: raw, config: config())
                    == .weightsNotConverted(
                        key: "embeddings.patch_embeddings.weight", shape: [32, 3, 16, 16]))
        }

        @Test("Checkpoints carrying the extra tensors of earlier versions still load")
        func legacyTensors() throws {
            var legacy = try weights()
            legacy["embeddings.mask_token"] = MLXArray.zeros([1, 1, 32])  // kept by 1.x Convert
            legacy["ropeEmbed.invFreq"] = MLXArray.zeros([4])  // saved from a 2.0.0 model
            let model = try load(weights: legacy, config: config())
            #expect(model.depth == 2)
        }

        @Test("Missing tensors are rejected")
        func missingTensors() throws {
            var incomplete = try weights()
            incomplete["embeddings.cls_token"] = nil
            incomplete["layer.1.norm1.weight"] = nil
            #expect(
                try loadError(weights: incomplete, config: config())
                    == .missingWeights(["embeddings.cls_token", "layer.1.norm1.weight"]))
        }

        @Test("Surplus tensors are rejected")
        func surplusTensors() throws {
            var surplus = try weights()
            surplus["classifier.weight"] = MLXArray.zeros([4, 64])
            #expect(
                try loadError(weights: surplus, config: config())
                    == .unexpectedWeights(["classifier.weight"]))
        }

        @Test("Wrong shapes are rejected")
        func wrongShape() throws {
            var wrong = try weights()
            wrong["norm.weight"] = MLXArray.ones([7])
            #expect(
                try loadError(weights: wrong, config: config())
                    == .shapeMismatch(key: "norm.weight", expected: [32], actual: [7]))
        }

        @Test("A config with fewer or more layers than the file is rejected")
        func layerCountMismatch() throws {
            let fewer = try loadError(
                weights: weights(), config: config { $0["num_hidden_layers"] = 1 })
            guard case .unexpectedWeights(let surplus) = fewer else {
                Issue.record("expected unexpectedWeights, got \(fewer)")
                return
            }
            #expect(surplus.allSatisfy { $0.hasPrefix("layer.1.") })

            let more = try loadError(
                weights: weights(), config: config { $0["num_hidden_layers"] = 3 })
            guard case .missingWeights(let missing) = more else {
                Issue.record("expected missingWeights, got \(more)")
                return
            }
            #expect(missing.allSatisfy { $0.hasPrefix("layer.2.") })
        }

        @Test("Config problems are reported by their config.json name")
        func invalidConfig() throws {
            func reason(_ edit: (inout [String: Any]) -> Void) throws -> String {
                let error = try loadError(weights: weights(), config: config(edit))
                guard case .invalidConfig(let reason) = error else {
                    Issue.record("expected invalidConfig, got \(error)")
                    return ""
                }
                return reason
            }

            #expect(try reason { $0["num_register_tokens"] = nil }.contains("num_register_tokens"))
            #expect(try reason { $0["hidden_size"] = "wide" }.contains("hidden_size"))
            #expect(try reason { $0["model_type"] = "dinov2" }.contains("dinov2"))
            #expect(try reason { $0["hidden_act"] = "tanh" }.contains("tanh"))
            #expect(try reason { $0["hidden_size"] = 30 }.contains("multiple"))
            #expect(try reason { $0["patch_size"] = 0 }.contains("patch_size"))
        }

        @Test("A config without hidden_act or model_type still loads")
        func optionalConfigKeys() throws {
            let model = try load(
                weights: weights(),
                config: config {
                    $0["hidden_act"] = nil
                    $0["model_type"] = nil
                })
            #expect(model.embedDim == 32)
        }
    }
}
