//
//  RoPEPositionEncoding.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import Foundation
import MLX
import MLXNN

public class RoPEPositionEncoding: Module {
    let dHead: Int
    let normalizeCoords: String
    let rescaleCoords: Float?
    let dtype: DType
    var periods: MLXArray

    public init(embedDim: Int,
                numHeads: Int,
                base: Float = 100.0,
                normalizeCoords: String = "separate",
                rescaleCoords: Float? = nil,
                dtype: DType = .float32)
    {
        precondition(embedDim % (4 * numHeads) == 0)

        dHead = embedDim / numHeads
        self.normalizeCoords = normalizeCoords
        self.rescaleCoords = rescaleCoords
        self.dtype = dtype

        let Dq = dHead / 4
        let idx = MLXArray(0 ..< Dq).asType(dtype)
        let exps = 2.0 * idx / Float(dHead / 2)
        periods = pow(MLXArray(base), exps).asType(dtype)

        super.init()
    }

    public func callAsFunction(H: Int, W: Int) -> (MLXArray, MLXArray) {
        var coordsH: MLXArray
        var coordsW: MLXArray

        switch normalizeCoords {
        case "max":
            let maxHW = Float(max(H, W))
            coordsH = MLXArray(stride(from: 0.5, to: Float(H), by: 1.0)).asType(dtype) / maxHW
            coordsW = MLXArray(stride(from: 0.5, to: Float(W), by: 1.0)).asType(dtype) / maxHW
        case "min":
            let minHW = Float(min(H, W))
            coordsH = MLXArray(stride(from: 0.5, to: Float(H), by: 1.0)).asType(dtype) / minHW
            coordsW = MLXArray(stride(from: 0.5, to: Float(W), by: 1.0)).asType(dtype) / minHW
        case "separate":
            coordsH = MLXArray(stride(from: 0.5, to: Float(H), by: 1.0)).asType(dtype) / Float(H)
            coordsW = MLXArray(stride(from: 0.5, to: Float(W), by: 1.0)).asType(dtype) / Float(W)
        default:
            fatalError("Unknown normalizeCoords: \(normalizeCoords)")
        }

        let coordsHGrid = coordsH[0..., .newAxis]
        let coordsWGrid = coordsW[.newAxis, 0...]
        let coords = stacked([
            broadcast(coordsHGrid, to: [H, W]),
            broadcast(coordsWGrid, to: [H, W]),
        ], axis: -1)

        var coords2D = coords.reshaped(-1, 2)
        coords2D = 2.0 * coords2D - 1.0

        if let rescale = rescaleCoords {
            coords2D = coords2D * rescale
        }

        let Dq = periods.dim(0)
        let HW = coords2D.dim(0)

        let coordsExpanded = coords2D[0..., 0..., .newAxis]
        let periodsExpanded = periods[.newAxis, .newAxis, 0...]

        var angles = (2.0 * Float.pi) * coordsExpanded / periodsExpanded
        angles = angles.reshaped(HW, 2 * Dq)
        angles = tiled(angles, repetitions: [1, 2])

        let cosAngles = cos(angles)
        let sinAngles = sin(angles)

        return (cosAngles, sinAngles)
    }
}
