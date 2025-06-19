import tensorrt as trt
import numpy as np
import cv2
import time
import common # 參考 NVIDIA 提供的 common.py 幫助函數，用於處理數據類型轉換和內存分配

# 設置 TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """
    從文件加載 TensorRT 引擎。
    """
    print(f"Loading engine from {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine, max_batch_size):
    """
    為 TensorRT 推理分配輸入/輸出緩衝區。
    """
    inputs = []
    outputs = []
    bindings = []
    stream = common.Stream() # 需要 common.py 中的 Stream 類
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 分配 host (CPU) 和 device (GPU) 內存
        host_mem = common.HostDeviceMem(size, dtype)
        device_mem = common.HostDeviceMem(size, dtype)
        bindings.append(int(device_mem.device))
        if engine.binding_is_input(binding):
            inputs.append(host_mem)
        else:
            outputs.append(host_mem)
    return inputs, outputs, bindings, stream

def preprocess_image(image_path, input_shape=(300, 300)):
    """
    預處理圖像以匹配模型輸入。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, input_shape)

    # 歸一化到 -1 到 1 (根據你模型訓練時的預處理方式調整)
    input_data = (image_resized / 127.5 - 1.0).astype(np.float32)

    # 將 HWC 轉為 CHW (如果模型是 ONNX 且輸入是 NCHW)
    # 對於 SSD-MobileNetV2，通常是 NCHW，但某些轉換工具可能輸出為 NHWC
    # 請根據你 ONNX 模型輸入的實際維度來判斷
    # 如果 Netron 顯示你的 ONNX 輸入是 (1, 3, 300, 300)，則需要 transpose:
    input_data = input_data.transpose((2, 0, 1)) # HWC to CHW

    return input_data

def postprocess_detections(output_data, img_width, img_height, score_threshold=0.5):
    """
    後處理模型輸出，解析邊界框、分數、類別。
    """
    # 這裡需要根據你的 SSD-MobileNetV2 具體輸出結構來調整
    # 通常輸出是 [1, num_detections, 4] for boxes, [1, num_detections] for scores, etc.
    # 你需要知道這些輸出在 TensorRT 輸出列表中的順序

    # 假設 TensorRT 輸出的順序是 boxes, scores, classes, num_detections
    # 並且它們已經被展平為一維數組
    # 你需要根據你的模型輸出節點在 ONNX 中的順序來映射
    # 例如，如果 outputs[0] 是 boxes, outputs[1] 是 scores, etc.
    num_detections = int(output_data[3][0])
    boxes = output_data[0][0][:num_detections]
    scores = output_data[1][0][:num_detections]
    classes = output_data[2][0][:num_detections]

    # 過濾低置信度檢測
    keep_indices = scores > score_threshold
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    classes = classes[keep_indices]

    # 將歸一化的框坐標轉換回原始圖像坐標
    # 注意：TensorFlow Object Detection API 的框坐標通常是 [ymin, xmin, ymax, xmax] 且歸一化到 [0, 1]
    # 你可能需要將它們轉換回 [xmin, ymin, xmax, ymax] 並乘以圖像寬高
    converted_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * img_width)
        xmax = int(xmax * img_width)
        ymin = int(ymin * img_height)
        ymax = int(ymax * img_height)
        converted_boxes.append([xmin, ymin, xmax, ymax])

    return converted_boxes, scores, classes

def main():
    engine_file_path = "ssd_mobilenet_v2.trt" # 替換為你的引擎文件路徑
    image_to_infer = "test_image.jpg"      # 替換為你要測試的圖片路徑
    score_threshold = 0.5                  # 調整置信度閾值

    # 加載引擎
    engine = load_engine(engine_file_path)
    if not engine:
        return

    # 創建執行上下文
    context = engine.create_execution_context()
    max_batch_size = 1 # 這裡要和構建引擎時的 batch_size 匹配

    # 分配緩衝區
    inputs, outputs, bindings, stream = allocate_buffers(engine, max_batch_size)

    # 讀取和預處理圖像
    original_image = cv2.imread(image_to_infer)
    img_height, img_width, _ = original_image.shape
    input_data = preprocess_image(image_to_infer)
    if input_data is None:
        return

    # 將預處理後的圖像數據複製到 host 輸入緩衝區
    inputs[0].host = input_data.flatten()

    # 執行推理
    print("Running inference...")
    start_time = time.time()
    # 執行推理 (需要 common.py 中的 do_inference 函數)
    # common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # 由於 common.py 依賴於 NVIDIA 的 TensorRT 範例，如果沒有，可能需要手動實現數據複製
    # 這裡簡單模擬：
    for inp in inputs:
        np.copyto(inp.device, inp.host) # 從 CPU 複製到 GPU
    context.execute_v2(bindings=bindings) # 執行推理
    for out in outputs:
        np.copyto(out.host, out.device) # 從 GPU 複製到 CPU

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # 後處理結果
    detected_boxes, detected_scores, detected_classes = postprocess_detections(
        [out.host for out in outputs], img_width, img_height, score_threshold
    )

    # 可視化結果 (例如，在圖像上繪製邊界框)
    output_image = original_image.copy()
    for i, box in enumerate(detected_boxes):
        xmin, ymin, xmax, ymax = box
        score = detected_scores[i]
        class_id = int(detected_classes[i]) # class_id 通常從0或1開始，根據你的label map調整

        # 假設你有名稱字典
        # class_name = label_map[class_id] # 你需要從 label_map.pbtxt 讀取類別名稱

        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"Class {class_id}: {score:.2f}" # 或 class_name
        cv2.putText(output_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"  Box: {box}, Score: {score:.2f}, Class: {class_id}")

    cv2.imwrite("output_image_detections.jpg", output_image)
    print("Inference complete. Results saved to output_image_detections.jpg")

if __name__ == "__main__":
    # 注意：你需要一個 'common.py' 文件來運行這個腳本，它通常在 TensorRT 範例中提供。
    # 如果你沒有 common.py，你需要手動實現 host/device 內存管理和數據傳輸。
    # 最簡單的方式是找到 TensorRT 的官方範例倉庫，其中通常包含 common.py。
    # 例如：https://github.com/NVIDIA/TensorRT/tree/main/samples/python
    print("Note: This script requires common.py for memory management.")
    print("Please ensure common.py is in the same directory or adjust imports.")
    main()