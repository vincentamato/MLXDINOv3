//
//  FFNLayers.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public class MLP: Module {
    @ModuleInfo var up_proj: Linear
    @ModuleInfo var down_proj: Linear
    let act: GELU

    public init(inFeatures: Int,
                hiddenFeatures: Int? = nil,
                outFeatures: Int? = nil,
                bias: Bool = true)
    {
        let hiddenFeatures = hiddenFeatures ?? inFeatures
        let outFeatures = outFeatures ?? inFeatures

        _up_proj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _down_proj.wrappedValue = Linear(hiddenFeatures, outFeatures, bias: bias)
        act = GELU()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(act(up_proj(x)))
    }
}

public class SwiGLUFFN: Module {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var up_proj: Linear
    @ModuleInfo var down_proj: Linear
    let act: SiLU

    public init(inFeatures: Int,
                hiddenFeatures: Int? = nil,
                outFeatures: Int? = nil,
                bias: Bool = true)
    {
        let hiddenFeatures = hiddenFeatures ?? inFeatures
        let outFeatures = outFeatures ?? inFeatures

        _gate_proj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _up_proj.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: bias)
        _down_proj.wrappedValue = Linear(hiddenFeatures, outFeatures, bias: bias)
        act = SiLU()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gate_proj(x)
        let up = up_proj(x)
        let hidden = act(gate) * up
        return down_proj(hidden)
    }
}
