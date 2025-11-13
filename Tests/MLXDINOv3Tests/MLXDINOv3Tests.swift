//
//  MLXDINOv3Tests.swift
//  MLXDINOv3Tests
//
//  Created by Vincent Amato on 10/29/25.
//

import AppKit
import Foundation
import Hub
import MLX
@testable import MLXDINOv3
import MLXNN
import Testing

actor TestResourceManager {
    static let shared = TestResourceManager()

    private var cachedPaths: (modelPath: URL, outputsPath: URL)?
    private var downloadTask: Task<(URL, URL), Error>?

    private init() {}

    func getResources() async throws -> (modelPath: URL, outputsPath: URL) {
        if let cached = cachedPaths {
            return cached
        }

        if let task = downloadTask {
            return try await task.value
        }

        let task = Task<(URL, URL), Error> {
            try await downloadResources()
        }
        downloadTask = task

        let paths = try await task.value
        cachedPaths = paths
        downloadTask = nil
        return paths
    }

    private func downloadResources() async throws -> (URL, URL) {
        guard let bundleURL = Bundle.module.resourceURL else {
            throw TestError.modelNotFound("""
            Could not find test bundle resource directory.

            To run tests, convert the model and place it in Tests/MLXDINOv3Tests/Resources/Model/:
              swift run Convert facebook/dinov3-vits16-pretrain-lvd1689m Tests/MLXDINOv3Tests/Resources/Model

            The directory should contain:
              - config.json
              - model.safetensors
            """)
        }

        let configPath = bundleURL.appendingPathComponent("config.json")
        let weightsPath = bundleURL.appendingPathComponent("model.safetensors")

        guard FileManager.default.fileExists(atPath: configPath.path),
              FileManager.default.fileExists(atPath: weightsPath.path)
        else {
            throw TestError.modelNotFound("""
            Model files not found in test bundle.

            To run tests, convert the model and place it in Tests/MLXDINOv3Tests/Resources/Model/:
              swift run Convert facebook/dinov3-vits16-pretrain-lvd1689m Tests/MLXDINOv3Tests/Resources/Model

            The directory should contain:
              - config.json
              - model.safetensors

            Expected location: \(bundleURL.path)
            """)
        }

        print("Using local model at \(bundleURL.path)")

        // Download PyTorch reference outputs from HuggingFace Hub
        let hub = HubApi()
        print("Downloading PyTorch reference outputs from vincentamato/dinov3-vits16-pretrain-lvd1689m-pt-outputs...")
        let outputsURL = try await hub.snapshot(from: "vincentamato/dinov3-vits16-pretrain-lvd1689m-pt-outputs")
        print("Reference outputs available at \(outputsURL.path)")

        return (bundleURL, outputsURL)
    }
}

@Suite("MLXDINOv3 Tests")
struct MLXDINOv3Tests {
    let absoluteTolerance: Float = 0.15
    let relativeTolerance: Float = 1e-2

    let minCosineSimilarityIntermediate: Float = 0.999
    let maxRelativeL2Intermediate: Float = 0.02

    let minCosineSimilarityAttention: Float = 0.999
    let maxRelativeL2Attention: Float = 0.03

    func ensureTestResources() async throws -> (modelPath: URL, outputsPath: URL) {
        try await TestResourceManager.shared.getResources()
    }

    func loadTestInputs(modelPath: URL, outputsPath _: URL) throws -> (image: NSImage, model: DinoVisionTransformer, inputs: MLXArray) {
        guard let imageURL = Bundle.module.url(forResource: "Image", withExtension: "jpg") else {
            throw TestError.imageLoadFailed
        }

        guard let image = NSImage(contentsOf: imageURL) else {
            print("Failed to load image from \(imageURL.path).")
            throw TestError.imageLoadFailed
        }

        let model = try loadPretrained(modelPath: modelPath.path)

        let processor = ImageProcessor(size: 224)
        let inputs = try processor(image)

        return (image, model, inputs)
    }

    func loadPyTorchOutput(filename: String, outputsPath: URL) throws -> [String: MLXArray] {
        let url = outputsPath.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TestError.missingTensor("Could not find \(filename) at \(url.path).")
        }
        return try MLX.loadArrays(url: url)
    }

    func arraysAreClose(_ a: MLXArray, _ b: MLXArray, rtol: Float, atol: Float) -> Bool {
        guard a.shape == b.shape else {
            print("Shape mismatch: \(a.shape) vs \(b.shape)")
            return false
        }

        let aCast = a.asType(.float32)
        let bCast = b.asType(.float32)

        let diff = abs(aCast - bCast)
        let threshold = MLXArray(atol) + MLXArray(rtol) * abs(bCast)
        let withinTolerance = MLX.lessEqual(diff, threshold)
        let allClose = withinTolerance.all()

        if !allClose.item(Bool.self) {
            let maxDiff = diff.max().item(Float.self)
            let meanDiff = mean(diff).item(Float.self)
            print("Max difference: \(maxDiff)")
            print("Mean difference: \(meanDiff)")
        }
        return allClose.item(Bool.self)
    }

    func arraysSimilar(_ a: MLXArray, _ b: MLXArray, minCosineSim: Float = 0.999, maxRelL2: Float = 0.02) -> Bool {
        guard a.shape == b.shape else {
            print("Shape mismatch: \(a.shape) vs \(b.shape)")
            return false
        }

        let aCast = a.asType(.float32).reshaped(-1)
        let bCast = b.asType(.float32).reshaped(-1)

        let dotProduct = (aCast * bCast).sum()
        let normA = sqrt((aCast * aCast).sum())
        let normB = sqrt((bCast * bCast).sum())
        let cosineSim = (dotProduct / (normA * normB)).item(Float.self)

        let diff = aCast - bCast
        let l2Diff = sqrt((diff * diff).sum())
        let relativeL2 = (l2Diff / normB).item(Float.self)

        let passCosineSim = cosineSim >= minCosineSim
        let passRelL2 = relativeL2 <= maxRelL2

        if !passCosineSim || !passRelL2 {
            print("Cosine similarity: \(cosineSim) (threshold: \(minCosineSim))")
            print("Relative L2 error: \(relativeL2) (threshold: \(maxRelL2))")
        }

        return passCosineSim && passRelL2
    }

    @Test("Pooler Output matches PyTorch")
    func testPoolerOutput() async throws {
        let (modelPath, outputsPath) = try await ensureTestResources()
        let (_, model, inputs) = try loadTestInputs(modelPath: modelPath, outputsPath: outputsPath)
        let outputs = model(inputs, outputHiddenStates: false, outputAttentions: false)

        let ptOutputs = try loadPyTorchOutput(filename: "pooler_output.safetensors", outputsPath: outputsPath)
        guard let ptPoolerOutput = ptOutputs["pooler_output"] else {
            throw TestError.missingTensor("pooler_output")
        }

        #expect(outputs.poolerOutput.shape == ptPoolerOutput.shape)
        let isClose = arraysAreClose(outputs.poolerOutput, ptPoolerOutput,
                                     rtol: relativeTolerance, atol: absoluteTolerance)
        #expect(isClose)
    }

    @Test("Last Hidden State matches PyTorch")
    func testLastHiddenState() async throws {
        let (modelPath, outputsPath) = try await ensureTestResources()
        let (_, model, inputs) = try loadTestInputs(modelPath: modelPath, outputsPath: outputsPath)
        let outputs = model(inputs, outputHiddenStates: false, outputAttentions: false)

        let ptOutputs = try loadPyTorchOutput(filename: "last_hidden_state.safetensors", outputsPath: outputsPath)
        guard let ptLastHidden = ptOutputs["last_hidden_state"] else {
            throw TestError.missingTensor("last_hidden_state")
        }

        #expect(outputs.lastHiddenState.shape == ptLastHidden.shape)
        let isClose = arraysAreClose(outputs.lastHiddenState, ptLastHidden,
                                     rtol: relativeTolerance, atol: absoluteTolerance)
        #expect(isClose)
    }

    @Test("Intermediate Hidden State (Layer 6) matches PyTorch")
    func hiddenState6() async throws {
        let (modelPath, outputsPath) = try await ensureTestResources()
        let (_, model, inputs) = try loadTestInputs(modelPath: modelPath, outputsPath: outputsPath)
        let outputs = model(inputs, outputHiddenStates: true, outputAttentions: false)

        let ptOutputs = try loadPyTorchOutput(filename: "hidden_state_6.safetensors", outputsPath: outputsPath)
        guard let ptHiddenState6 = ptOutputs["hidden_state_6"] else {
            throw TestError.missingTensor("hidden_state_6")
        }

        guard let hiddenStates = outputs.hiddenStates,
              let mlxHiddenState6 = hiddenStates["hidden_state_6"]
        else {
            throw TestError.missingOutput("hidden_state_6")
        }

        #expect(mlxHiddenState6.shape == ptHiddenState6.shape)
        let isSimilar = arraysSimilar(mlxHiddenState6, ptHiddenState6,
                                      minCosineSim: minCosineSimilarityIntermediate, maxRelL2: maxRelativeL2Intermediate)
        #expect(isSimilar)
    }

    @Test("Attention Weights (Layer 6) match PyTorch")
    func attention6() async throws {
        let (modelPath, outputsPath) = try await ensureTestResources()
        let (_, model, inputs) = try loadTestInputs(modelPath: modelPath, outputsPath: outputsPath)
        let outputs = model(inputs, outputHiddenStates: false, outputAttentions: true)

        let ptOutputs = try loadPyTorchOutput(filename: "attention_6.safetensors", outputsPath: outputsPath)
        guard let ptAttention6 = ptOutputs["attention_6"] else {
            throw TestError.missingTensor("attention_6")
        }

        guard let attentions = outputs.attentions, attentions.count > 6 else {
            throw TestError.missingOutput("attentions")
        }
        let mlxAttention6 = attentions[6]

        #expect(mlxAttention6.shape == ptAttention6.shape)
        let isSimilar = arraysSimilar(mlxAttention6, ptAttention6,
                                      minCosineSim: minCosineSimilarityAttention, maxRelL2: maxRelativeL2Attention)
        #expect(isSimilar)
    }

    @Test("Output shapes are correct")
    func outputShapes() async throws {
        let (modelPath, outputsPath) = try await ensureTestResources()
        let (_, model, inputs) = try loadTestInputs(modelPath: modelPath, outputsPath: outputsPath)
        let outputs = model(inputs, outputHiddenStates: true, outputAttentions: true)

        let B = 1, D = 384, N = 201, H = 6, numLayers = 12
        #expect(outputs.poolerOutput.shape == [B, D])
        #expect(outputs.lastHiddenState.shape == [B, N, D])

        if let hiddenStates = outputs.hiddenStates {
            #expect(hiddenStates.keys.count(where: { $0.starts(with: "hidden_state_") }) == numLayers + 1)
        }

        if let attentions = outputs.attentions {
            #expect(attentions.count == numLayers)
            for (idx, attn) in attentions.enumerated() {
                #expect(attn.shape == [B, H, N, N], "Attention layer \(idx) shape incorrect")
            }
        }
    }
}

enum TestError: Error {
    case imageLoadFailed
    case missingTensor(String)
    case missingOutput(String)
    case invalidOutput(String)
    case modelNotFound(String)
}
