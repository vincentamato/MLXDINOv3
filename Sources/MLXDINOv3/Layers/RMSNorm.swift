//
//  RMSNorm.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    private func norm(_ x: MLXArray) -> MLXArray {
        let mean = (x * x).mean(axes: [-1], keepDims: true)
        return x * rsqrt(mean + eps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = norm(x)
        return output * weight
    }
}
