import MLX
import MLXNN

final class Embeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: Conv2d
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "register_tokens") var registerTokens: MLXArray

    init(
        patchSize: Int,
        inChannels: Int,
        embedDim: Int,
        numRegisterTokens: Int
    ) {
        _patchEmbeddings.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize))
        _clsToken.wrappedValue = zeros([1, 1, embedDim])
        _registerTokens.wrappedValue = zeros([1, numRegisterTokens, embedDim])
    }
}
