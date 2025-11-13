// swift-tools-version: 6.0
import PackageDescription

let mlxDeps: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
]

let package = Package(
    name: "MLXDINOv3",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MLXDINOv3",
            targets: ["MLXDINOv3"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.1")
    ],
    targets: [
        .target(
            name: "MLXDINOv3",
            dependencies: mlxDeps,
            path: "Sources/MLXDINOv3"
        ),
        .executableTarget(
            name: "Convert",
            dependencies: mlxDeps + [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Sources/Convert"
        ),
        .testTarget(
            name: "MLXDINOv3Tests",
            dependencies: [
                "MLXDINOv3",
                .product(name: "Hub", package: "swift-transformers")
            ],
            path: "Tests/MLXDINOv3Tests",
            resources: [
                .process("Resources")
            ]
        )
    ]
)
