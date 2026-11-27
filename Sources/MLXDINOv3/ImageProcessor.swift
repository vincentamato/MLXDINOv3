import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UIKit)
    import UIKit

    public typealias PlatformImage = UIImage
#elseif canImport(AppKit)
    import AppKit

    public typealias PlatformImage = NSImage
#endif

/// Per-channel mean and standard deviation used to normalize pixel values.
public struct NormalizationStats: Sendable {
    public let mean: [Float]
    public let std: [Float]

    public init(mean: [Float], std: [Float]) {
        precondition(mean.count == 3 && std.count == 3, "Expect 3-channel mean/std")
        self.mean = mean
        self.std = std
    }

    /// ImageNet statistics, used by the DINOv3 checkpoints trained on web images (LVD-1689M).
    public static let imageNet = NormalizationStats(
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225]
    )
}

/// Turns an image into the input tensor of a ``DinoVisionTransformer``.
///
/// The image is resized to `size` x `size` with bilinear resampling, scaled to 0...1 and
/// normalized. The result is a `float32` array of shape `[1, size, size, 3]` (channels last).
///
/// Things to know:
/// - The image is stretched to a square; the aspect ratio is not preserved. This matches the
///   Hugging Face DINOv3 image processor.
/// - The orientation of a `UIImage`, or the EXIF orientation of an `NSImage` loaded from a file,
///   is applied, so the model sees the image upright.
/// - Transparent pixels are composited onto black.
/// - See ``ColorProfile`` for how wide-gamut images (such as iPhone photos) are handled.
/// - Resizing is a port of Pillow's `Image.resize(..., BILINEAR)` and is bit-identical to
///   Pillow 9.5 through 12.x for 8-bit RGB input. (Pillow 13 sometimes runs the vertical pass
///   first, which can change a value by one level.) Hugging Face's processor resizes in floating
///   point instead, which also differs from this by less than one level.
public final class ImageProcessor: Sendable {
    public enum Error: Swift.Error {
        /// The image has no pixels, or its pixel data could not be read.
        case invalidImage
    }

    /// What to do with an image's color profile.
    public enum ColorProfile: Sendable {
        /// Convert to sRGB, so the model sees the sRGB values of the colors in the image. A
        /// Display P3 photo therefore gets different numbers from the ones stored in the file.
        case convertToSRGB
        /// Use the stored values unchanged, which is what Pillow and torchvision do. Choose this
        /// to reproduce the output of a Python pipeline. Applies to RGB images; other color
        /// models (grayscale, CMYK) are always converted.
        case ignore
    }

    private let targetSize: Int
    private let norm: NormalizationStats
    private let colorProfile: ColorProfile

    /// - Parameters:
    ///   - size: side length of the square output, in pixels. Use a multiple of the model's
    ///     patch size; a remainder is dropped by the model.
    ///   - normalizationStats: per-channel mean and standard deviation.
    ///   - colorProfile: how an embedded color profile is treated.
    public init(
        size: Int = 224,
        normalizationStats: NormalizationStats = .imageNet,
        colorProfile: ColorProfile = .convertToSRGB
    ) {
        precondition(size > 0, "size must be positive")
        targetSize = size
        norm = normalizationStats
        self.colorProfile = colorProfile
    }

    /// Preprocesses a `UIImage` or `NSImage`.
    public func callAsFunction(_ image: PlatformImage) throws -> MLXArray {
        guard let (cgImage, orientation) = orientedCGImage(from: image) else {
            throw Error.invalidImage
        }
        return try self(cgImage, orientation: orientation)
    }

    /// Preprocesses a `CGImage`.
    ///
    /// - Parameter orientation: how the stored pixels must be turned to display the image
    ///   upright, for example the EXIF orientation of the file the image came from.
    public func callAsFunction(
        _ image: CGImage, orientation: CGImagePropertyOrientation = .up
    ) throws -> MLXArray {
        let resized = try resizedPixels(image, orientation: orientation)
        return expandedDimensions(
            MLXArray(normalize(resized), [targetSize, targetSize, 3]), axis: 0)
    }

    /// Number of source pixels drawn at a time: 16 MB of RGBA.
    static let bandPixelBudget = 4 << 20

    /// Resizes to `targetSize` x `targetSize` and returns 8-bit RGB, row-major.
    ///
    /// The source is decoded in horizontal bands that are reduced to the target width straight
    /// away, so a 48-megapixel photo does not need a second and third full-size copy in memory.
    func resizedPixels(
        _ image: CGImage,
        orientation: CGImagePropertyOrientation = .up,
        rowsPerBand: Int? = nil
    ) throws -> [UInt8] {
        let swapsAxes: Bool
        switch orientation {
        case .left, .leftMirrored, .right, .rightMirrored: swapsAxes = true
        default: swapsAxes = false
        }
        let srcW = swapsAxes ? image.height : image.width
        let srcH = swapsAxes ? image.width : image.height
        guard srcW > 0, srcH > 0 else { throw Error.invalidImage }

        let deviceRGB = CGColorSpaceCreateDeviceRGB()
        var source = image
        if colorProfile == .ignore, let untagged = image.copy(colorSpace: deviceRGB) {
            source = untagged
        }

        let bandRows = max(1, min(srcH, rowsPerBand ?? Self.bandPixelBudget / srcW))
        let bitmapInfo =
            CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard
            let context = CGContext(
                data: nil, width: srcW, height: bandRows,
                bitsPerComponent: 8, bytesPerRow: srcW * 4,
                space: deviceRGB, bitmapInfo: bitmapInfo),
            let data = context.data?.assumingMemoryBound(to: UInt8.self)
        else { throw Error.invalidImage }
        context.interpolationQuality = .none

        let upright = orientationTransform(
            orientation, width: CGFloat(image.width), height: CGFloat(image.height))
        let imageRect = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let bandRect = CGRect(x: 0, y: 0, width: srcW, height: bandRows)

        let hCoeffs = PILResampler.precomputeCoefficients(inSize: srcW, outSize: targetSize)
        var temp = [UInt8](repeating: 0, count: targetSize * srcH * 3)

        var firstRow = 0
        while firstRow < srcH {
            let rows = min(bandRows, srcH - firstRow)
            context.clear(bandRect)
            context.saveGState()
            // Put row `firstRow` of the upright image on the top row of the bitmap.
            context.translateBy(x: 0, y: CGFloat(bandRows - srcH + firstRow))
            context.concatenate(upright)
            context.draw(source, in: imageRect)
            context.restoreGState()

            PILResampler.resampleHorizontal(
                UnsafeBufferPointer(start: data, count: srcW * bandRows * 4),
                srcW: srcW, rows: rows, into: &temp, firstRow: firstRow,
                dstW: targetSize, coeffs: hCoeffs)
            firstRow += rows
        }

        let vCoeffs = PILResampler.precomputeCoefficients(inSize: srcH, outSize: targetSize)
        return PILResampler.resampleVertical(
            temp, srcW: targetSize, dstH: targetSize, coeffs: vCoeffs)
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

/// The bitmap behind a platform image and how to turn it upright.
private func orientedCGImage(
    from image: PlatformImage
) -> (CGImage, CGImagePropertyOrientation)? {
    #if canImport(UIKit)
        if let cgImage = image.cgImage {
            let orientation: CGImagePropertyOrientation
            switch image.imageOrientation {
            case .up: orientation = .up
            case .upMirrored: orientation = .upMirrored
            case .down: orientation = .down
            case .downMirrored: orientation = .downMirrored
            case .left: orientation = .left
            case .leftMirrored: orientation = .leftMirrored
            case .right: orientation = .right
            case .rightMirrored: orientation = .rightMirrored
            @unknown default: orientation = .up
            }
            return (cgImage, orientation)
        }
        // No bitmap backing (for example a CIImage-based image): have UIKit render it upright.
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        let rendered = UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
        return rendered.cgImage.map { ($0, .up) }
    #else
        // AppKit has already applied the EXIF orientation.
        var rect = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil).map { ($0, .up) }
    #endif
}

/// Maps the stored image, drawn in `(0, 0, width, height)`, onto its upright position in a
/// bottom-left-origin context.
private func orientationTransform(
    _ orientation: CGImagePropertyOrientation, width w: CGFloat, height h: CGFloat
) -> CGAffineTransform {
    switch orientation {
    case .up: return .identity
    case .upMirrored: return CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
    case .down: return CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
    case .downMirrored: return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
    case .leftMirrored: return CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: h, ty: w)
    case .right: return CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: w)
    case .rightMirrored: return CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
    case .left: return CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: h, ty: 0)
    }
}

// Port of Pillow's bilinear resampling (horizontal pass, then vertical pass).
// Reference: https://github.com/python-pillow/Pillow/blob/12.0.0/src/libImaging/Resample.c
// Pillow's license is reproduced in THIRD_PARTY_NOTICES.md.
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

    /// Resamples the first `rows` rows of an RGBA buffer `srcW` pixels wide to `dstW` pixels and
    /// writes them as RGB into `dst`, starting at row `firstRow`.
    static func resampleHorizontal(
        _ src: UnsafeBufferPointer<UInt8>, srcW: Int, rows: Int,
        into dst: inout [UInt8], firstRow: Int,
        dstW: Int, coeffs: Coefficients
    ) {
        dst.withUnsafeMutableBufferPointer { dstBuf in
            coeffs.weights.withUnsafeBufferPointer { wBuf in
                coeffs.bounds.withUnsafeBufferPointer { bounds in
                    for y in 0 ..< rows {
                        let srcRowBase = y * srcW * 4
                        let dstRowBase = (firstRow + y) * dstW * 3

                        for x in 0 ..< dstW {
                            let xmin = bounds[x * 2]
                            let count = bounds[x * 2 + 1]
                            let kBase = x * coeffs.ksize

                            var r = fixedPointHalf
                            var g = fixedPointHalf
                            var b = fixedPointHalf

                            for k in 0 ..< count {
                                let si = srcRowBase + (k + xmin) * 4
                                let w = wBuf[kBase + k]
                                r += Int32(src[si]) * w
                                g += Int32(src[si + 1]) * w
                                b += Int32(src[si + 2]) * w
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
}
