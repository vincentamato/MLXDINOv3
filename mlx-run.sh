#!/bin/sh

# Adapted from https://github.com/ml-explore/mlx-swift-examples/blob/main/mlx-run

set -e

if [ "$#" -lt 1 ]; then
    echo "usage: mlx-run [--debug] <tool-name> arguments"
    exit 1
fi

CONFIGURATION=Release
if [ "$1" = "--debug" ]; then
    CONFIGURATION=Debug
    shift
fi
if [ "$1" = "--release" ]; then
    shift
fi

COMMAND="$1"
shift

DERIVED_DATA=".build/xcode"

xcodebuild build \
    -scheme "$COMMAND" \
    -configuration "$CONFIGURATION" \
    -destination "platform=macOS" \
    -derivedDataPath "$DERIVED_DATA" \
    -quiet

BUILD_DIR="$DERIVED_DATA/Build/Products/$CONFIGURATION"

if [ -f "$BUILD_DIR/$COMMAND" ]; then
    export DYLD_FRAMEWORK_PATH="$BUILD_DIR/PackageFrameworks:$BUILD_DIR"
    exec "$BUILD_DIR/$COMMAND" "$@"
else
    echo "$BUILD_DIR/$COMMAND does not exist"
    exit 1
fi