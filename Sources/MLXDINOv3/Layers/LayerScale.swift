//
//  LayerScale.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public class LayerScale: Module {
    @ModuleInfo var lambda1: MLXArray

    public init(dim: Int, initValues: Float = 1e-5) {
        _lambda1.wrappedValue = MLXArray.full([dim], values: MLXArray(initValues))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * lambda1
    }
}
