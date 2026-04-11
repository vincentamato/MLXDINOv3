import AppKit
import Hub
import MLX
import Testing

@testable import MLXDINOv3

actor TestResourceManager {
    static let shared = TestResourceManager()

    private var loaded: (model: DinoVisionTransformer, inputs: MLXArray, outputsPath: URL)?
    private var task: Task<Void, Error>?

    private init() {}

    func withResources<T>(
        _ body: (DinoVisionTransformer, MLXArray, URL) throws -> T
    ) async throws -> T {
        if task == nil {
            task = Task { self.loaded = try await self.load() }
        }
        guard let task else {
            throw TestError.resourceLoadFailed
        }
        try await task.value
        guard let r = loaded else {
            throw TestError.resourceLoadFailed
        }
        return try body(r.model, r.inputs, r.outputsPath)
    }

    private func load() async throws -> (DinoVisionTransformer, MLXArray, URL) {
        guard let bundleURL = Bundle.module.resourceURL,
            FileManager.default.fileExists(
                atPath: bundleURL.appendingPathComponent("config.json").path),
            FileManager.default.fileExists(
                atPath: bundleURL.appendingPathComponent("model.safetensors").path)
        else {
            throw TestError.modelNotFound(
                """
                Model not found in test bundle. Run:
                  ./mlx-run.sh Convert facebook/dinov3-vits16-pretrain-lvd1689m Tests/MLXDINOv3Tests/Resources
                """)
        }

        let hub = HubApi()
        let outputsURL = try await hub.snapshot(
            from: "vincentamato/dinov3-vits16-pretrain-lvd1689m-pt-outputs"
        )

        let model = try DinoVisionTransformer.loadPretrained(from: bundleURL.path)

        guard let imageURL = Bundle.module.url(forResource: "image", withExtension: "jpg"),
            let image = NSImage(contentsOf: imageURL)
        else {
            throw TestError.imageLoadFailed
        }
        let inputs = try ImageProcessor()(image)

        return (model, inputs, outputsURL)
    }
}

@Suite("MLXDINOv3 Tests")
struct MLXDINOv3Tests {
    private func loadReference(_ filename: String, key: String, from outputsPath: URL) throws
        -> MLXArray
    {
        let url = outputsPath.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TestError.missingTensor("Could not find \(filename) at \(url.path).")
        }
        let arrays = try loadArrays(url: url)
        guard let tensor = arrays[key] else {
            throw TestError.missingTensor(key)
        }
        return tensor
    }

    private func assertSimilar(
        _ a: MLXArray, _ b: MLXArray,
        minCosineSim: Float = 0.999,
        maxRelL2: Float = 0.02
    ) throws {
        #expect(a.shape == b.shape)

        let aFlat = a.asType(.float32).reshaped(-1)
        let bFlat = b.asType(.float32).reshaped(-1)

        let dot = (aFlat * bFlat).sum()
        let normA = sqrt((aFlat * aFlat).sum())
        let normB = sqrt((bFlat * bFlat).sum())
        let cosineSim = (dot / (normA * normB)).item(Float.self)

        let diff = aFlat - bFlat
        let relL2 = (sqrt((diff * diff).sum()) / normB).item(Float.self)

        #expect(
            cosineSim >= minCosineSim,
            "Cosine similarity \(cosineSim) < \(minCosineSim)")
        #expect(
            relL2 <= maxRelL2,
            "Relative L2 \(relL2) > \(maxRelL2)")
    }

    @Test("Output shapes are correct")
    func outputShapes() async throws {
        try await TestResourceManager.shared.withResources { model, inputs, _ in
            let outputs = model(inputs, outputHiddenStates: true)

            let B = inputs.dim(0)
            let D = outputs.clsToken.dim(-1)
            let N = outputs.lastHiddenState.dim(1)
            let P = outputs.patchTokens.dim(1)

            #expect(outputs.clsToken.shape == [B, D])
            #expect(outputs.patchTokens.shape == [B, P, D])
            #expect(outputs.lastHiddenState.shape == [B, N, D])
            #expect(N == 1 + 4 + P)

            if let hiddenStates = outputs.hiddenStates {
                #expect(hiddenStates.count == model.layers.count + 1)
                for state in hiddenStates {
                    #expect(state.shape == [B, N, D])
                }
            }
        }
    }

    @Test("Last hidden state matches PyTorch")
    func lastHiddenState() async throws {
        try await TestResourceManager.shared.withResources { model, inputs, outputsPath in
            let ref = try loadReference(
                "last_hidden_state.safetensors", key: "last_hidden_state", from: outputsPath)
            try assertSimilar(model(inputs).lastHiddenState, ref)
        }
    }

    @Test("CLS token matches PyTorch")
    func clsToken() async throws {
        try await TestResourceManager.shared.withResources { model, inputs, outputsPath in
            let ref = try loadReference(
                "pooler_output.safetensors", key: "pooler_output", from: outputsPath)
            try assertSimilar(model(inputs).clsToken, ref)
        }
    }

    @Test("Intermediate hidden state (layer 6) matches PyTorch")
    func hiddenState6() async throws {
        try await TestResourceManager.shared.withResources { model, inputs, outputsPath in
            let ref = try loadReference(
                "hidden_state_6.safetensors", key: "hidden_state_6", from: outputsPath)
            let outputs = model(inputs, outputHiddenStates: true)

            guard let hiddenStates = outputs.hiddenStates, hiddenStates.count > 6 else {
                throw TestError.missingOutput("hidden_state_6")
            }

            try assertSimilar(hiddenStates[6], ref)
        }
    }
}

enum TestError: Error {
    case imageLoadFailed
    case missingTensor(String)
    case missingOutput(String)
    case modelNotFound(String)
    case resourceLoadFailed
}
