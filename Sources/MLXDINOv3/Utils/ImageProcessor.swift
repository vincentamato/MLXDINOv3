//
//  ImageProcessor.swift
//  MLXDINOv3
//
//  Created by Vincent Amato on 10/29/25.
//

import CoreGraphics
import Foundation
import MLX
import MLXNN

#if canImport(UIKit)
    import UIKit

    public typealias PlatformImage = UIImage
#endif

#if canImport(AppKit)
    import AppKit

    public typealias PlatformImage = NSImage
#endif

public struct NormalizationStats: Sendable {
    public let mean: [Float]
    public let std: [Float]
    public init(mean: [Float], std: [Float]) {
        precondition(mean.count == 3 && std.count == 3, "Expect 3-channel mean/std")
        self.mean = mean; self.std = std
    }

    public static let imageNet = NormalizationStats(mean: [0.485, 0.456, 0.406],
                                                    std: [0.229, 0.224, 0.225])
}

public enum PreprocessError: Error, CustomStringConvertible {
    case invalidImage
    case pixelDataUnavailable

    public var description: String {
        switch self {
        case .invalidImage: "Invalid image / cannot get CGImage backing."
        case .pixelDataUnavailable: "Failed to obtain resized pixel buffer."
        }
    }
}

public final class ImageProcessor {
    public let targetSize: Int
    public let norm: NormalizationStats

    public init(size: Int = 224,
                normalizationStats: NormalizationStats = .imageNet)
    {
        targetSize = size
        norm = normalizationStats
    }

    public func preprocessHWC(_ image: PlatformImage) throws -> [Float] {
        guard let cg = cgImage(from: image) else { throw PreprocessError.invalidImage }

        guard let (srcPixels, srcW, srcH) = extractRGBPixels(from: cg) else {
            throw PreprocessError.pixelDataUnavailable
        }

        let resizedPixels = bilinearResizePIL(srcPixels,
                                              srcW: srcW,
                                              srcH: srcH,
                                              dstW: targetSize,
                                              dstH: targetSize)

        let w = targetSize, h = targetSize
        var out = [Float](repeating: 0, count: h * w * 3)
        let mean = norm.mean, std = norm.std

        for i in 0 ..< (h * w * 3) {
            let pixelValue = Float(resizedPixels[i]) / 255.0
            let channel = i % 3
            out[i] = (pixelValue - mean[channel]) / std[channel]
        }

        return out
    }

    public func preprocessNHWC(_ image: PlatformImage) throws -> [Float] {
        try preprocessHWC(image)
    }

    public func preprocessHWC_MLX(_ image: PlatformImage) throws -> MLXArray {
        let hwc = try preprocessHWC(image)
        return MLXArray(hwc, [targetSize, targetSize, 3])
    }

    public func preprocessNHWC_MLX(_ image: PlatformImage) throws -> MLXArray {
        let hwc = try preprocessHWC(image)
        let arr = MLXArray(hwc, [targetSize, targetSize, 3])
        return expandedDimensions(arr, axis: 0)
    }

    public func callAsFunction(_ image: PlatformImage) throws -> MLXArray {
        try preprocessNHWC_MLX(image)
    }
}

private func cgImage(from image: PlatformImage) -> CGImage? {
    #if canImport(UIKit)
        return image.cgImage
    #else
        var rect = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    #endif
}

private func bilinearFilter(_ x: Double) -> Double {
    let absX = abs(x)
    if absX < 1.0 {
        return 1.0 - absX
    }
    return 0.0
}

private struct ResampleCoeffs {
    let ksize: Int
    let bounds: [Int]
    let kk: [Double]
}

private func precomputeCoeffs(inSize: Int,
                              in0: Double,
                              in1: Double,
                              outSize: Int,
                              filterSupport: Double) -> ResampleCoeffs
{
    let scale = (in1 - in0) / Double(outSize)
    let filterScale = max(scale, 1.0)
    let support = filterSupport * filterScale
    let ksize = Int(ceil(support)) * 2 + 1

    var kk = [Double](repeating: 0, count: outSize * ksize)
    var bounds = [Int](repeating: 0, count: outSize * 2)

    for xx in 0 ..< outSize {
        let center = in0 + (Double(xx) + 0.5) * scale
        let ss = 1.0 / filterScale

        var xmin = Int((center - support + 0.5).rounded(.towardZero))
        if xmin < 0 {
            xmin = 0
        }

        var xmax = Int((center + support + 0.5).rounded(.towardZero))
        if xmax > inSize {
            xmax = inSize
        }
        xmax -= xmin

        var ww = 0.0
        let kBase = xx * ksize

        for x in 0 ..< xmax {
            let w = bilinearFilter((Double(x + xmin) - center + 0.5) * ss)
            kk[kBase + x] = w
            ww += w
        }

        if ww != 0.0 {
            for x in 0 ..< xmax {
                kk[kBase + x] /= ww
            }
        }

        bounds[xx * 2 + 0] = xmin
        bounds[xx * 2 + 1] = xmax
    }

    return ResampleCoeffs(ksize: ksize, bounds: bounds, kk: kk)
}

private func normalizeCoeffs8bpc(_ coeffs: ResampleCoeffs, outSize: Int) -> [Int32] {
    let precisionBits = 22
    let scale = Double(1 << precisionBits)
    var normalized = [Int32](repeating: 0, count: outSize * coeffs.ksize)

    for i in 0 ..< (outSize * coeffs.ksize) {
        let val = coeffs.kk[i]
        if val < 0 {
            normalized[i] = Int32((-0.5 + val * scale).rounded(.towardZero))
        } else {
            normalized[i] = Int32((0.5 + val * scale).rounded(.towardZero))
        }
    }

    return normalized
}

@inline(__always)
private func clip8(_ value: Int32, precisionBits: Int) -> UInt8 {
    let shifted = value >> precisionBits
    if shifted <= 0 { return 0 }
    if shifted >= 255 { return 255 }
    return UInt8(shifted)
}

private func resampleHorizontal(src: [UInt8],
                                srcW: Int,
                                srcH: Int,
                                dstW: Int,
                                coeffs: ResampleCoeffs,
                                normalizedKK: [Int32]) -> [UInt8]
{
    let precisionBits = 22
    var dst = [UInt8](repeating: 0, count: dstW * srcH * 3)

    for yy in 0 ..< srcH {
        for xx in 0 ..< dstW {
            let xmin = coeffs.bounds[xx * 2 + 0]
            let xmax = coeffs.bounds[xx * 2 + 1]
            let kBase = xx * coeffs.ksize

            var ss0: Int32 = 1 << (precisionBits - 1)
            var ss1: Int32 = 1 << (precisionBits - 1)
            var ss2: Int32 = 1 << (precisionBits - 1)

            for x in 0 ..< xmax {
                let srcIdx = (yy * srcW + (x + xmin)) * 3
                let k = normalizedKK[kBase + x]
                ss0 += Int32(src[srcIdx + 0]) * k
                ss1 += Int32(src[srcIdx + 1]) * k
                ss2 += Int32(src[srcIdx + 2]) * k
            }

            let dstIdx = (yy * dstW + xx) * 3
            dst[dstIdx + 0] = clip8(ss0, precisionBits: precisionBits)
            dst[dstIdx + 1] = clip8(ss1, precisionBits: precisionBits)
            dst[dstIdx + 2] = clip8(ss2, precisionBits: precisionBits)
        }
    }

    return dst
}

private func resampleVertical(src: [UInt8],
                              srcW: Int,
                              srcH _: Int,
                              dstH: Int,
                              coeffs: ResampleCoeffs,
                              normalizedKK: [Int32]) -> [UInt8]
{
    let precisionBits = 22
    var dst = [UInt8](repeating: 0, count: srcW * dstH * 3)

    for yy in 0 ..< dstH {
        let ymin = coeffs.bounds[yy * 2 + 0]
        let ymax = coeffs.bounds[yy * 2 + 1]
        let kBase = yy * coeffs.ksize

        for xx in 0 ..< srcW {
            var ss0: Int32 = 1 << (precisionBits - 1)
            var ss1: Int32 = 1 << (precisionBits - 1)
            var ss2: Int32 = 1 << (precisionBits - 1)

            for y in 0 ..< ymax {
                let srcIdx = ((y + ymin) * srcW + xx) * 3
                let k = normalizedKK[kBase + y]
                ss0 += Int32(src[srcIdx + 0]) * k
                ss1 += Int32(src[srcIdx + 1]) * k
                ss2 += Int32(src[srcIdx + 2]) * k
            }

            let dstIdx = (yy * srcW + xx) * 3
            dst[dstIdx + 0] = clip8(ss0, precisionBits: precisionBits)
            dst[dstIdx + 1] = clip8(ss1, precisionBits: precisionBits)
            dst[dstIdx + 2] = clip8(ss2, precisionBits: precisionBits)
        }
    }

    return dst
}

private func bilinearResizePIL(_ src: [UInt8],
                               srcW: Int,
                               srcH: Int,
                               dstW: Int,
                               dstH: Int) -> [UInt8]
{
    let bilinearSupport = 1.0

    let horizCoeffs = precomputeCoeffs(inSize: srcW,
                                       in0: 0.0,
                                       in1: Double(srcW),
                                       outSize: dstW,
                                       filterSupport: bilinearSupport)

    let horizNormalized = normalizeCoeffs8bpc(horizCoeffs, outSize: dstW)
    let tempImage = resampleHorizontal(src: src,
                                       srcW: srcW,
                                       srcH: srcH,
                                       dstW: dstW,
                                       coeffs: horizCoeffs,
                                       normalizedKK: horizNormalized)

    let vertCoeffs = precomputeCoeffs(inSize: srcH,
                                      in0: 0.0,
                                      in1: Double(srcH),
                                      outSize: dstH,
                                      filterSupport: bilinearSupport)

    let vertNormalized = normalizeCoeffs8bpc(vertCoeffs, outSize: dstH)
    let finalImage = resampleVertical(src: tempImage,
                                      srcW: dstW,
                                      srcH: srcH,
                                      dstH: dstH,
                                      coeffs: vertCoeffs,
                                      normalizedKK: vertNormalized)

    return finalImage
}

private func extractRGBPixels(from cgImage: CGImage) -> (pixels: [UInt8], width: Int, height: Int)? {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let bitsPerComponent = 8

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

    guard let context = CGContext(data: nil,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: bitsPerComponent,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: bitmapInfo) else { return nil }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let dataPtr = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }

    var rgbPixels = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0 ..< height {
        let row = dataPtr.advanced(by: y * bytesPerRow)
        for x in 0 ..< width {
            let rgbaBase = x * 4
            let rgbBase = (y * width + x) * 3
            rgbPixels[rgbBase + 0] = row[rgbaBase + 0]
            rgbPixels[rgbBase + 1] = row[rgbaBase + 1]
            rgbPixels[rgbBase + 2] = row[rgbaBase + 2]
        }
    }

    return (rgbPixels, width, height)
}
