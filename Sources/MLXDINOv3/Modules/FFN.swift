import MLX
import MLXNN

final class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let act: GELU

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool) {
        _upProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _downProj.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: bias)
        act = GELU()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(act(upProj(x)))
    }
}

final class SwiGLUFFN: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let act: SiLU

    init(inFeatures: Int, hiddenFeatures: Int, bias: Bool) {
        _gateProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _upProj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _downProj.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: bias)
        act = SiLU()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        let hidden = act(gate) * up
        return downProj(hidden)
    }
}
