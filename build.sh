#!/bin/bash

# Qdrant 构建脚本
# 使用方法: ./build.sh [release|debug]

set -e

BUILD_TYPE=${1:-release}

echo "=========================================="
echo "Qdrant 构建脚本"
echo "=========================================="

# 检查 Rust 是否安装
if ! command -v cargo &> /dev/null; then
    echo "错误: 未找到 cargo 命令"
    echo ""
    echo "请先安装 Rust:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  或者访问: https://www.rust-lang.org/tools/install"
    echo ""
    echo "安装后，请运行: source ~/.cargo/env"
    exit 1
fi

echo "Rust 版本:"
rustc --version
cargo --version
echo ""

# 检查 protoc 是否安装（用于编译 Protocol Buffers）
if ! command -v protoc &> /dev/null; then
    echo "警告: 未找到 protoc 命令"
    echo "Protocol Buffers 编译器可能需要安装"
    echo "macOS: brew install protobuf"
    echo "Linux: sudo apt-get install protobuf-compiler"
    echo ""
fi

# 设置构建类型
if [ "$BUILD_TYPE" = "release" ]; then
    echo "构建类型: Release (优化版本)"
    BUILD_FLAG="--release"
    OUTPUT_DIR="target/release"
else
    echo "构建类型: Debug (开发版本)"
    BUILD_FLAG=""
    OUTPUT_DIR="target/debug"
fi

echo ""
echo "开始构建..."
echo "这可能需要一些时间，请耐心等待..."
echo ""

# 执行构建
cargo build $BUILD_FLAG

# 检查构建结果
if [ -f "$OUTPUT_DIR/qdrant" ]; then
    echo ""
    echo "=========================================="
    echo "构建成功！"
    echo "=========================================="
    echo "可执行文件位置: $OUTPUT_DIR/qdrant"
    echo ""
    ls -lh "$OUTPUT_DIR/qdrant"
    echo ""
    echo "运行方式:"
    echo "  $OUTPUT_DIR/qdrant"
    echo ""
    echo "或者使用配置文件:"
    echo "  $OUTPUT_DIR/qdrant --config-path config/config.yaml"
else
    echo ""
    echo "=========================================="
    echo "构建失败！"
    echo "=========================================="
    echo "请检查上面的错误信息"
    exit 1
fi
