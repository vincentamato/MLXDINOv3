import Foundation
import MLX
import Testing

@testable import Convert
@testable import MLXDINOv3

extension AllTests {
    /// The Convert tool's weight mapping, on a PyTorch state dict from the fixtures.
    @Suite("Convert")
    struct ConvertTests {
        let directory = Fixtures.model("plain")

        func pytorchWeights() throws -> [String: MLXArray] {
            try loadArrays(url: directory.appendingPathComponent("pytorch_model.safetensors"))
        }

        func convertedWeights() throws -> [String: MLXArray] {
            try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        }

        func config() throws -> [String: Any] {
            try loadConfig(from: directory)
        }

        func expectSame(_ actual: [String: MLXArray], _ expected: [String: MLXArray]) {
            #expect(Set(actual.keys) == Set(expected.keys))
            for (key, value) in expected {
                guard let other = actual[key], other.shape == value.shape else {
                    Issue.record("\(key): missing or wrong shape")
                    continue
                }
                #expect(arrayEqual(other, value).item(Bool.self), "\(key) differs")
            }
        }

        @Test("A PyTorch state dict converts to the layout the model loads")
        func convertsStateDict() throws {
            let (weights, notes) = try convertWeights(pytorchWeights(), config: config())
            expectSame(weights, try convertedWeights())
            #expect(notes.isEmpty)
        }

        @Test("Prefixed tensor names are converted, and reported")
        func prefixedNames() throws {
            var prefixed = Dictionary(
                uniqueKeysWithValues: try pytorchWeights().map { ("backbone." + $0.key, $0.value) })
            prefixed["classifier.weight"] = MLXArray.zeros([4, 64])

            let (kept, notes) = try convertWeights(prefixed, config: config())
            #expect(kept["backbone.embeddings.patch_embeddings.weight"]?.shape == [32, 16, 16, 3])
            #expect(kept["backbone.embeddings.mask_token"] == nil)
            #expect(kept["classifier.weight"] != nil)
            #expect(notes.contains { $0.contains("--strip-prefix backbone.") })

            let (stripped, _) = try convertWeights(
                prefixed, config: config(), stripPrefix: "backbone.")
            expectSame(stripped, try convertedWeights())

            #expect(throws: ConversionError.prefixNotFound("encoder.")) {
                try convertWeights(prefixed, config: config(), stripPrefix: "encoder.")
            }
        }

        @Test("An already converted checkpoint is left alone")
        func alreadyConverted() throws {
            let (weights, notes) = try convertWeights(convertedWeights(), config: config())
            expectSame(weights, try convertedWeights())
            #expect(notes.contains { $0.contains("already in MLX layout") })
        }

        @Test("Checkpoints that are not DINOv3 ViTs are rejected")
        func rejectsOtherModels() throws {
            var wrongShape = try pytorchWeights()
            wrongShape["embeddings.patch_embeddings.weight"] = MLXArray.zeros([32, 3, 14, 14])
            #expect(
                throws: ConversionError.unexpectedShape(
                    key: "embeddings.patch_embeddings.weight", shape: [32, 3, 14, 14])
            ) {
                try convertWeights(wrongShape, config: config())
            }

            var noPatches = try pytorchWeights()
            noPatches["embeddings.patch_embeddings.weight"] = nil
            #expect(throws: ConversionError.missingTensor("embeddings.patch_embeddings.weight")) {
                try convertWeights(noPatches, config: config())
            }

            let scratch = try TemporaryDirectory()
            try Data(#"{"model_type": "dinov2"}"#.utf8)
                .write(to: scratch.url.appendingPathComponent("config.json"))
            #expect(throws: ConversionError.unsupportedModelType(modelType: "dinov2")) {
                try loadConfig(from: scratch.url)
            }
        }

        @Test("A local checkpoint directory converts into a model that matches PyTorch")
        func convertsLocalDirectory() async throws {
            let source = try TemporaryDirectory()
            try FileManager.default.copyItem(
                at: directory.appendingPathComponent("config.json"),
                to: source.url.appendingPathComponent("config.json"))
            try FileManager.default.copyItem(
                at: directory.appendingPathComponent("pytorch_model.safetensors"),
                to: source.url.appendingPathComponent("model.safetensors"))
            let output = try TemporaryDirectory()

            try await convert(source: source.url.path, outputDir: output.url.path)

            let model = try DinoVisionTransformer.loadPretrained(from: output.url.path)
            let io = try loadArrays(url: directory.appendingPathComponent("io.safetensors"))
            expectClose(
                model(try #require(io["batch.pixel_values"])).lastHiddenState,
                try #require(io["batch.last_hidden_state"]))
        }
    }
}
