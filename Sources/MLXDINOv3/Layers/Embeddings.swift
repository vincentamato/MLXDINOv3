//
//  Embeddings.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/30/25.
//

import Foundation
import MLX
import MLXNN

public class Embeddings: Module {
    @ModuleInfo var patch_embeddings: PatchEmbed
    @ModuleInfo var cls_token: MLXArray
    @ModuleInfo var register_tokens: MLXArray?
    @ModuleInfo var mask_token: MLXArray

    public init(imgSize: Int,
                patchSize: Int,
                inChannels: Int,
                embedDim: Int,
                layerNormEps: Float,
                numRegisterTokens: Int)
    {
        _patch_embeddings.wrappedValue = PatchEmbed(imgSize: imgSize,
                                                    patchSize: patchSize,
                                                    inChannels: inChannels,
                                                    embedDim: embedDim,
                                                    usePatchNorm: false,
                                                    flattenEmbedding: false,
                                                    layerNormEps: layerNormEps)

        _cls_token.wrappedValue = MLXArray.zeros([1, 1, embedDim])

        if numRegisterTokens > 0 {
            _register_tokens.wrappedValue = MLXArray.zeros([1, numRegisterTokens, embedDim])
        }

        _mask_token.wrappedValue = MLXArray.zeros([1, embedDim])
    }
}
