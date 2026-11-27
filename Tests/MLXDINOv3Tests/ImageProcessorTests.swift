import AppKit
import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing

@testable import MLXDINOv3

extension AllTests {
    /// ImageProcessor against what Pillow 12 produces for the same files.
    @Suite("ImageProcessor")
    struct ImageProcessorTests {
        /// Leaves pixel values as 0...1 so they can be compared with 8-bit references.
        static let unnormalized = NormalizationStats(mean: [0, 0, 0], std: [1, 1, 1])

        func loadImage(_ name: String) throws -> NSImage {
            try #require(NSImage(contentsOf: Fixtures.image("\(name).png")))
        }

        /// The 8-bit pixels behind an unnormalized processor output.
        func pixels(_ output: MLXArray) -> MLXArray {
            round(output[0] * 255)
        }

        func cgImage(
            rgb: MLXArray, colorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        ) throws -> CGImage {
            let (height, width) = (rgb.dim(0), rgb.dim(1))
            let bytes = rgb.asType(.uint8).asArray(UInt8.self)
            let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
            return try #require(
                CGImage(
                    width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                    bytesPerRow: width * 3, space: colorSpace,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                    provider: provider, decode: nil, shouldInterpolate: false,
                    intent: .defaultIntent))
        }

        @Test(
            "Resizing is bit-identical to Pillow",
            arguments: [
                ("noise", 32), ("noise", 64), ("noise", 37),  // 60x40: down and up
                ("photo", 32), ("photo", 224),  // 99x148 portrait: down and up
                ("gray", 32),
            ])
        func matchesPillow(_ name: String, _ size: Int) throws {
            let expected = try #require(try Fixtures.expectedPixels()["\(name).\(size)"])
            let processor = ImageProcessor(size: size, normalizationStats: Self.unnormalized)
            let output = try processor(loadImage(name))
            #expect(output.shape == [1, size, size, 3])
            #expect(maxPixelDifference(pixels(output), expected) == 0)
        }

        @Test("Default output is normalized float32, channels last")
        func defaultNormalization() throws {
            let expected = try #require(try Fixtures.expectedPixels()["photo.224"])
            let output = try ImageProcessor()(loadImage("photo"))
            #expect(output.shape == [1, 224, 224, 3])
            #expect(output.dtype == .float32)

            let mean = MLXArray([0.485, 0.456, 0.406] as [Float])
            let std = MLXArray([0.229, 0.224, 0.225] as [Float])
            expectClose(output[0], (expected.asType(.float32) / 255 - mean) / std, tolerance: 1e-5)
        }

        @Test("Transparent pixels are composited onto black")
        func transparency() throws {
            let expected = try #require(try Fixtures.expectedPixels()["alpha.over_black.32"])
            let processor = ImageProcessor(size: 32, normalizationStats: Self.unnormalized)
            #expect(maxPixelDifference(pixels(try processor(loadImage("alpha"))), expected) == 0)
        }

        @Test("Decoding in bands does not change the result", arguments: [1, 7, 147, 148, 1000])
        func bands(_ rowsPerBand: Int) throws {
            let expected = try Fixtures.expectedPixels()
            let image = try cgImage(rgb: try #require(expected["photo.raw"]))
            let processor = ImageProcessor(size: 32)
            let resized = try processor.resizedPixels(image, rowsPerBand: rowsPerBand)
            #expect(
                maxPixelDifference(
                    MLXArray(resized, [32, 32, 3]), try #require(expected["photo.32"]))
                    == 0)
        }

        @Test("Every EXIF orientation is turned upright", arguments: 1 ... 8, [nil, 5])
        func orientation(_ exifValue: Int, _ rowsPerBand: Int?) throws {
            let expected = try Fixtures.expectedPixels()
            let stored = try cgImage(rgb: try #require(expected["orientation.raw"]))
            let orientation = try #require(CGImagePropertyOrientation(rawValue: UInt32(exifValue)))
            let processor = ImageProcessor(size: 16, normalizationStats: Self.unnormalized)
            let want = try #require(expected["orientation.\(exifValue).16"])

            let resized = try processor.resizedPixels(
                stored, orientation: orientation, rowsPerBand: rowsPerBand)
            #expect(maxPixelDifference(MLXArray(resized, [16, 16, 3]), want) == 0)

            if rowsPerBand == nil {
                let output = try processor(stored, orientation: orientation)
                #expect(maxPixelDifference(pixels(output), want) == 0)
            }
        }

        @Test("Color profiles are converted to sRGB unless told to ignore them")
        func colorProfiles() throws {
            let expected = try Fixtures.expectedPixels()
            let raw = try #require(expected["photo.raw"])
            let want = try #require(expected["photo.32"])
            let displayP3 = try #require(CGColorSpace(name: CGColorSpace.displayP3))
            let tagged = try cgImage(rgb: raw, colorSpace: displayP3)

            // Pillow ignores the profile, so `.ignore` must reproduce it exactly.
            let ignoring = ImageProcessor(
                size: 32, normalizationStats: Self.unnormalized, colorProfile: .ignore)
            #expect(maxPixelDifference(pixels(try ignoring(tagged)), want) == 0)

            // The default converts Display P3 to sRGB, which moves saturated colors.
            let converting = ImageProcessor(size: 32, normalizationStats: Self.unnormalized)
            #expect(maxPixelDifference(pixels(try converting(tagged)), want) > 4)

            // An sRGB image is the same either way.
            let untagged = try cgImage(rgb: raw)
            #expect(maxPixelDifference(pixels(try converting(untagged)), want) == 0)
            #expect(maxPixelDifference(pixels(try ignoring(untagged)), want) == 0)
        }

        @Test("An image without pixels is an error")
        func emptyImage() {
            #expect(throws: ImageProcessor.Error.invalidImage) {
                try ImageProcessor()(NSImage())
            }
        }
    }
}
