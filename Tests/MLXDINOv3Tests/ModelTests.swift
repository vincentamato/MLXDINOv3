import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXDINOv3

extension AllTests {
    /// The forward pass against Hugging Face's PyTorch implementation, on random-weight models.
    @Suite("Model")
    struct ModelTests {
        @Test(
            "Outputs match PyTorch",
            arguments: ["plain", "gated", "no_registers", "gated_gelu", "plain_relu"])
        func matchesPyTorch(_ name: String) throws {
            let directory = Fixtures.model(name)
            let model = try DinoVisionTransformer.loadPretrained(from: directory.path)
            let io = try loadArrays(url: directory.appendingPathComponent("io.safetensors"))
            let inputs = io.keys.filter { $0.hasSuffix(".pixel_values") }
                .map { String($0.dropLast(".pixel_values".count)) }
            #expect(!inputs.isEmpty)

            for input in inputs {
                let pixelValues = try #require(io["\(input).pixel_values"])
                let output = model(pixelValues, outputHiddenStates: true)

                expectClose(
                    output.lastHiddenState, try #require(io["\(input).last_hidden_state"]),
                    "\(name)/\(input) lastHiddenState")
                expectClose(
                    output.clsToken, try #require(io["\(input).pooler_output"]),
                    "\(name)/\(input) clsToken")

                let hiddenStates = try #require(output.hiddenStates)
                #expect(hiddenStates.count == model.depth + 1)
                for (index, state) in hiddenStates.enumerated() {
                    expectClose(
                        state, try #require(io["\(input).hidden_state_\(index)"]),
                        "\(name)/\(input) hiddenStates[\(index)]")
                }

                let (batch, tokens) = (pixelValues.dim(0), output.lastHiddenState.dim(1))
                let patches =
                    (pixelValues.dim(1) / model.patchSize) * (pixelValues.dim(2) / model.patchSize)
                #expect(tokens == 1 + model.numRegisterTokens + patches)
                #expect(output.clsToken.shape == [batch, model.embedDim])
                #expect(output.patchTokens.shape == [batch, patches, model.embedDim])
                #expect(model(pixelValues).hiddenStates == nil)
            }
        }

        @Test("Parameters are exactly the tensors of the checkpoint")
        func parametersMatchCheckpoint() throws {
            let directory = Fixtures.model("plain")
            let model = try DinoVisionTransformer.loadPretrained(from: directory.path)
            let file = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))

            let parameters = Set(model.parameters().flattened().map { $0.0 })
            let trainable = Set(model.trainableParameters().flattened().map { $0.0 })
            #expect(parameters == Set(file.keys))
            #expect(trainable == Set(file.keys))
        }

        @Test("Model exposes its dimensions")
        func dimensions() throws {
            let model = try DinoVisionTransformer.loadPretrained(
                from: Fixtures.model("plain").path)
            #expect(model.patchSize == 16)
            #expect(model.inChannels == 3)
            #expect(model.embedDim == 32)
            #expect(model.depth == 2)
            #expect(model.numHeads == 2)
            #expect(model.numRegisterTokens == 4)
        }

        @Test(
            "Half-precision weights run in half precision", arguments: [DType.float16, .bfloat16])
        func halfPrecision(_ dtype: DType) throws {
            let directory = Fixtures.model("plain")
            let weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
            let io = try loadArrays(url: directory.appendingPathComponent("io.safetensors"))

            let scratch = try TemporaryDirectory()
            try scratch.writeModel(
                weights: weights.mapValues { $0.asType(dtype) },
                config: Data(contentsOf: directory.appendingPathComponent("config.json")))
            let model = try DinoVisionTransformer.loadPretrained(from: scratch.url.path)

            // ImageProcessor produces float32; the model casts it to the weights' precision.
            let output = model(try #require(io["batch.pixel_values"]))
            #expect(output.lastHiddenState.dtype == dtype)
            expectClose(
                output.lastHiddenState, try #require(io["batch.last_hidden_state"]),
                tolerance: 2e-2, "\(dtype) lastHiddenState")
        }
    }
}
