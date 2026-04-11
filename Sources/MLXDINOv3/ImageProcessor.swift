import CoreGraphics
import Foundation
import MLX

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
        self.mean = mean
        self.std = std
    }

    public static let imageNet = NormalizationStats(
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225]
    )
}

public final class ImageProcessor: Sendable {
    public enum Error: Swift.Error {
        case invalidImage
    }

    private let targetSize: Int
    private let norm: NormalizationStats

    public init(
        size: Int = 224,
        normalizationStats: NormalizationStats = .imageNet
    ) {
        targetSize = size
        norm = normalizationStats
    }

    public func callAsFunction(_ image: PlatformImage) throws -> MLXArray {
        guard let cg = cgImage(from: image),
            let (srcPixels, srcW, srcH) = extractRGBPixels(from: cg)
        else {
            throw Error.invalidImage
        }

        let resized = PILResampler.bilinearResize(
            srcPixels, srcW: srcW, srcH: srcH, dstW: targetSize, dstH: targetSize)

        let normalized = normalize(resized)
        return expandedDimensions(MLXArray(normalized, [targetSize, targetSize, 3]), axis: 0)
    }

    private func normalize(_ pixels: [UInt8]) -> [Float] {
        let scales = zip(norm.mean, norm.std).map { 1.0 / (255.0 * $1) }
        let offsets = zip(norm.mean, norm.std).map { -$0 / $1 }

        let count = pixels.count
        var out = [Float](repeating: 0, count: count)

        for i in 0 ..< count {
            let c = i % 3
            out[i] = Float(pixels[i]) * scales[c] + offsets[c]
        }

        return out
    }
}

private func cgImage(from image: PlatformImage) -> CGImage? {
    #if canImport(UIKit)
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }.cgImage
    #else
        var rect = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    #endif
}

private func extractRGBPixels(from cgImage: CGImage) -> (pixels: [UInt8], width: Int, height: Int)?
{
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerRow = width * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo =
        CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

    guard
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo
        )
    else { return nil }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let data = context.data?.assumingMemoryBound(to: UInt8.self) else { return nil }

    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for i in 0 ..< (width * height) {
        rgb[i * 3] = data[i * 4]
        rgb[i * 3 + 1] = data[i * 4 + 1]
        rgb[i * 3 + 2] = data[i * 4 + 2]
    }

    return (rgb, width, height)
}

// Port of Pillow's bilinear resampling
// Reference: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
private enum PILResampler {
    static let precisionBits = 22
    static let fixedPointHalf: Int32 = 1 << 21

    struct Coefficients {
        let ksize: Int
        let bounds: [Int]
        let weights: [Int32]
    }

    static func precomputeCoefficients(inSize: Int, outSize: Int) -> Coefficients {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = filterScale
        let ksize = Int(ceil(support)) * 2 + 1

        var floatWeights = [Double](repeating: 0, count: outSize * ksize)
        var bounds = [Int](repeating: 0, count: outSize * 2)

        for out in 0 ..< outSize {
            let center = (Double(out) + 0.5) * scale
            let ss = 1.0 / filterScale

            let xmin = max(0, Int((center - support + 0.5).rounded(.towardZero)))
            var xmax = min(inSize, Int((center + support + 0.5).rounded(.towardZero)))
            xmax -= xmin

            let base = out * ksize
            var weightSum = 0.0

            for x in 0 ..< xmax {
                let w = max(0, 1.0 - abs((Double(x + xmin) - center + 0.5) * ss))
                floatWeights[base + x] = w
                weightSum += w
            }

            if weightSum != 0.0 {
                for x in 0 ..< xmax {
                    floatWeights[base + x] /= weightSum
                }
            }

            bounds[out * 2] = xmin
            bounds[out * 2 + 1] = xmax
        }

        let fpScale = Double(1 << precisionBits)
        let intWeights = floatWeights.map { v -> Int32 in
            Int32(
                v < 0
                    ? (-0.5 + v * fpScale).rounded(.towardZero)
                    : (0.5 + v * fpScale).rounded(.towardZero))
        }

        return Coefficients(ksize: ksize, bounds: bounds, weights: intWeights)
    }

    @inline(__always)
    static func clip8(_ value: Int32) -> UInt8 {
        let shifted = value >> precisionBits
        if shifted <= 0 { return 0 }
        if shifted >= 255 { return 255 }
        return UInt8(shifted)
    }

    static func resampleHorizontal(
        _ src: [UInt8], srcW: Int, srcH: Int,
        dstW: Int, coeffs: Coefficients
    ) -> [UInt8] {
        var dst = [UInt8](repeating: 0, count: dstW * srcH * 3)

        src.withUnsafeBufferPointer { srcBuf in
            dst.withUnsafeMutableBufferPointer { dstBuf in
                coeffs.weights.withUnsafeBufferPointer { wBuf in
                    for y in 0 ..< srcH {
                        let srcRowBase = y * srcW * 3
                        let dstRowBase = y * dstW * 3

                        for x in 0 ..< dstW {
                            let xmin = coeffs.bounds[x * 2]
                            let count = coeffs.bounds[x * 2 + 1]
                            let kBase = x * coeffs.ksize

                            var r = fixedPointHalf
                            var g = fixedPointHalf
                            var b = fixedPointHalf

                            for k in 0 ..< count {
                                let si = srcRowBase + (k + xmin) * 3
                                let w = wBuf[kBase + k]
                                r += Int32(srcBuf[si]) * w
                                g += Int32(srcBuf[si + 1]) * w
                                b += Int32(srcBuf[si + 2]) * w
                            }

                            let di = dstRowBase + x * 3
                            dstBuf[di] = clip8(r)
                            dstBuf[di + 1] = clip8(g)
                            dstBuf[di + 2] = clip8(b)
                        }
                    }
                }
            }
        }

        return dst
    }

    static func resampleVertical(
        _ src: [UInt8], srcW: Int,
        dstH: Int, coeffs: Coefficients
    ) -> [UInt8] {
        var dst = [UInt8](repeating: 0, count: srcW * dstH * 3)

        src.withUnsafeBufferPointer { srcBuf in
            dst.withUnsafeMutableBufferPointer { dstBuf in
                coeffs.weights.withUnsafeBufferPointer { wBuf in
                    for y in 0 ..< dstH {
                        let ymin = coeffs.bounds[y * 2]
                        let count = coeffs.bounds[y * 2 + 1]
                        let kBase = y * coeffs.ksize
                        let dstRowBase = y * srcW * 3

                        for x in 0 ..< srcW {
                            var r = fixedPointHalf
                            var g = fixedPointHalf
                            var b = fixedPointHalf

                            for k in 0 ..< count {
                                let si = ((k + ymin) * srcW + x) * 3
                                let w = wBuf[kBase + k]
                                r += Int32(srcBuf[si]) * w
                                g += Int32(srcBuf[si + 1]) * w
                                b += Int32(srcBuf[si + 2]) * w
                            }

                            let di = dstRowBase + x * 3
                            dstBuf[di] = clip8(r)
                            dstBuf[di + 1] = clip8(g)
                            dstBuf[di + 2] = clip8(b)
                        }
                    }
                }
            }
        }

        return dst
    }

    static func bilinearResize(
        _ src: [UInt8], srcW: Int, srcH: Int,
        dstW: Int, dstH: Int
    ) -> [UInt8] {
        let hCoeffs = precomputeCoefficients(inSize: srcW, outSize: dstW)
        let temp = resampleHorizontal(
            src, srcW: srcW, srcH: srcH,
            dstW: dstW, coeffs: hCoeffs)

        let vCoeffs = precomputeCoefficients(inSize: srcH, outSize: dstH)
        return resampleVertical(temp, srcW: dstW, dstH: dstH, coeffs: vCoeffs)
    }
}
