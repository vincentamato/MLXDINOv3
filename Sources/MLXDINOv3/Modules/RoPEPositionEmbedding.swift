import MLX

/// Rotary position embedding for the patch tokens.
///
/// This is deliberately not a `Module`. The frequencies are fixed, so they must stay out of
/// `parameters()`: otherwise an optimizer would train them and they would be written into saved
/// checkpoints under a key no other DINOv3 implementation knows.
struct RoPEPositionEmbedding {
    private let invFreq: MLXArray

    init(
        embedDim: Int,
        numHeads: Int,
        base: Float = 100.0
    ) {
        precondition(
            numHeads > 0 && embedDim % (4 * numHeads) == 0,
            "embedDim (\(embedDim)) must be a multiple of 4 * numHeads (\(numHeads))")

        let headDim = embedDim / numHeads
        let Dq = headDim / 4
        let exponents = MLXArray(0 ..< Dq).asType(.float32) * (4.0 / Float(headDim))
        invFreq = 1.0 / pow(MLXArray(base), exponents)
    }

    /// Returns `(cos, sin)`, each of shape `[H * W, headDim]`.
    func callAsFunction(H: Int, W: Int, dtype: DType = .float32) -> (MLXArray, MLXArray) {
        let coordsH = MLXArray(stride(from: 0.5, to: Float(H), by: 1.0)) / Float(H)
        let coordsW = MLXArray(stride(from: 0.5, to: Float(W), by: 1.0)) / Float(W)

        let coordsHGrid = coordsH[0..., .newAxis]
        let coordsWGrid = coordsW[.newAxis, 0...]
        let coords = stacked(
            [
                broadcast(coordsHGrid, to: [H, W]),
                broadcast(coordsWGrid, to: [H, W]),
            ], axis: -1)

        var coords2D = coords.reshaped(-1, 2)
        coords2D = 2.0 * coords2D - 1.0

        let Dq = invFreq.dim(0)
        let HW = coords2D.dim(0)

        let coordsExpanded = coords2D[0..., 0..., .newAxis]
        let invFreqExpanded = invFreq[.newAxis, .newAxis, 0...]

        var angles = (2.0 * Float.pi) * coordsExpanded * invFreqExpanded
        angles = angles.reshaped(HW, 2 * Dq)
        angles = tiled(angles, repetitions: [1, 2])

        return (cos(angles).asType(dtype), sin(angles).asType(dtype))
    }
}
