import MLX
import MLXNN

final class LayerScale: Module {
    @ModuleInfo var lambda1: MLXArray

    init(dim: Int, initValues: Float) {
        _lambda1.wrappedValue = full([dim], values: initValues)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * lambda1
    }
}
