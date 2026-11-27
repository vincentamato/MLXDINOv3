import MLX
import MLXNN

typealias Activation = DinoVisionTransformer.Activation

final class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let act: Activation

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool, activation: Activation = .gelu) {
        _upProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _downProj.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: bias)
        act = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(act(upProj(x)))
    }
}

final class SwiGLUFFN: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let act: Activation

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool, activation: Activation = .silu) {
        _gateProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _upProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _downProj.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: bias)
        act = activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        let hidden = act(gate) * up
        return downProj(hidden)
    }
}
