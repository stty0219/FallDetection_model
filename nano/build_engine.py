import tensorrt as trt
import onnx
import argparse
import os

# 設置 TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 設置日誌級別為警告及以上

def build_engine(onnx_file_path, engine_file_path, precision='FP16', max_batch_size=1):
    """
    從 ONNX 模型構建 TensorRT 引擎。
    """
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 1. 解析 ONNX 模型
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not os.path.exists(onnx_file_path):
        print(f"Error: ONNX file not found at {onnx_file_path}")
        return None
    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Error parsing ONNX model:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("ONNX parsing complete.")

    # 設置輸入維度
    # 通常對於檢測模型，輸入名稱為 'image_tensor'，形狀為 (1, 300, 300, 3) 或 (1, 3, 300, 300)
    # 你需要根據你的模型實際輸入名稱和數據格式來調整
    # 在這裡，我們假設輸入節點已經被 ONNX Parser 正確識別
    # network.get_input(0).shape = [max_batch_size, 3, 300, 300] # 如果你的模型是 CHW 格式
    network.get_input(0).shape = [max_batch_size, 300, 300, 3] # 如果你的模型是 HWC 格式

    # 設置構建配置
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB 工作空間，可根據 Jetson Nano 內存調整
    builder.max_batch_size = max_batch_size # deprecated in TRT 8.x, but still good to set for older versions or clarity

    if precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building engine with FP16 precision.")
    elif precision == 'INT8':
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 需要校準器 (Calibrator)，這會更複雜，這裡不詳述
        # 例如：config.int8_calibrator = common.DataloaderCalibrator(...)
        print("Building engine with INT8 precision (requires calibrator).")
    else:
        print("Building engine with FP32 precision.")

    # 構建引擎
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Error: Engine building failed.")
        return None
    print("Engine building complete.")

    # 保存引擎
    print(f"Saving engine to {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Engine saved successfully.")
    return engine

def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--engine_path", type=str, default="model.trt", help="Path to save the TensorRT engine file.")
    parser.add_argument("--precision", type=str, default="FP16", choices=["FP32", "FP16", "INT8"], help="Precision to build the engine (FP32, FP16, INT8).")
    parser.add_argument("--batch_size", type=int, default=1, help="Max batch size for the engine.")
    args = parser.parse_args()

    build_engine(args.onnx_path, args.engine_path, args.precision, args.batch_size)

if __name__ == "__main__":
    main()