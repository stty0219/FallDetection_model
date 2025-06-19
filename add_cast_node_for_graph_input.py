import onnx
from onnx import helper
from onnx import TensorProto

# 定義您的原始模型檔案名稱
original_model_path = "new_model_v1/model_v1_simplified.onnx" # 請將此替換為您的實際 ONNX 檔案路徑
output_model_path = "new_model_v1/model_v1_float32_noCast.onnx" # 新的輸出檔案名

# 定義原始輸入的名稱 (根據您的 Netron 截圖，通常是 'input_tensor:0' 或類似名稱)
input_to_modify = 'input_tensor:0' # 請確認您的模型輸入名稱

try:
    # 載入原始 ONNX 模型
    print(f"載入原始模型： {original_model_path}")
    orig_model = onnx.load_model(original_model_path)
    new_model = onnx.load_model(original_model_path) # 建立一個副本進行修改

    # 找到原始輸入的 TensorProto 資訊
    original_input_info = None
    for input_proto in orig_model.graph.input:
        if input_proto.name == input_to_modify:
            original_input_info = input_proto
            break

    if original_input_info is None:
        raise ValueError(f"在模型中找不到名為 '{input_to_modify}' 的輸入。請檢查輸入名稱是否正確。")

    # 1. 修改圖的輸入
    # 清空現有的輸入，然後添加一個新的 FLOAT32 類型的輸入
    del new_model.graph.input[:]
    
    # 使用原始輸入的名稱，但將其類型更改為 FLOAT32
    # 形狀應該與原始輸入相同
    new_model.graph.input.append(
        helper.make_tensor_value_info(
            input_to_modify, # 使用原始輸入名稱
            TensorProto.FLOAT, # 新的資料類型為 FLOAT32
            # 複製原始輸入的形狀 (假設是動態形狀，可以直接複製 dim_param)
            [d.dim_value if d.HasField('dim_value') else d.dim_param for d in original_input_info.type.tensor_type.shape.dim]
        )
    )

    # 2. 圖的節點部分保持不變
    # 由於我們沒有插入 Cast 節點，所以新模型直接使用原始模型的節點列表
    # 這一步可以省略，因為我們沒有修改 nodes 列表

    # 3. 檢查和儲存修改後的模型
    print("檢查修改後的模型...")
    onnx.checker.check_model(new_model)
    print("模型檢查成功！")

    print(f"儲存修改後的模型到： {output_model_path}")
    onnx.save_model(new_model, output_model_path)
    print("模型修改並儲存完成。")
    print(f"現在，您的模型 '{output_model_path}' 期望 Float32 輸入，且沒有內部 Cast 節點。")
    print("\n重要：在運行推論時，請確保您的影像資料在傳遞給此模型之前，已轉換為 Float32 並進行了適當的正規化！")

except Exception as e:
    print(f"處理模型時發生錯誤： {e}")

