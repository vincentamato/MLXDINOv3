//
//  PatchEmbed.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public final class PatchEmbed: Module {
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    var norm: LayerNorm?
    let imgSize: (Int, Int)
    let patchSize: (Int, Int)
    let patchesResolution: (Int, Int)
    let numPatches: Int
    let flattenEmbedding: Bool
    let inChannels: Int
    let embedDim: Int

    public init(imgSize: Int = 224,
                patchSize: Int = 16,
                inChannels: Int = 3,
                embedDim: Int = 768,
                usePatchNorm: Bool = false,
                flattenEmbedding: Bool = true,
                layerNormEps: Float = 1e-05)
    {
        self.imgSize = (imgSize, imgSize)
        self.patchSize = (patchSize, patchSize)
        self.flattenEmbedding = flattenEmbedding
        self.inChannels = inChannels
        self.embedDim = embedDim

        let H = imgSize / patchSize
        let W = imgSize / patchSize
        patchesResolution = (H, W)
        numPatches = H * W

        let scale = sqrt(1.0 / Float(inChannels * patchSize * patchSize))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale,
                                                 high: scale,
                                                 [embedDim, patchSize, patchSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([embedDim])

        norm = usePatchNorm ? LayerNorm(dimensions: embedDim, eps: layerNormEps) : nil
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = conv2d(xIn,
                       weight,
                       stride: [patchSize.0, patchSize.1],
                       padding: [0, 0])
        x = x + bias.reshaped(1, 1, 1, embedDim)
        let (B, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        x = x.reshaped(B, H * W, C)

        if let norm {
            x = norm(x)
        }

        if !flattenEmbedding {
            x = x.reshaped(B, H, W, C)
        }

        return x
    }
}
