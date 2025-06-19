import onnx
from onnx import helper
from onnx import TensorProto

# 定義您的原始模型檔案名稱
original_model_path = "new_model_v1/model_v1_simplified.onnx" # 請將此替換為您的實際 ONNX 檔案路徑
output_model_path = "your_model_float32_input.onnx"

# 定義原始輸入的名稱 (根據您的 Netron 截圖，通常是 'input_tensor:0' 或類似名稱)
input_to_replace = 'input_tensor:0' # 請確認您的模型輸入名稱

try:
    # 載入原始 ONNX 模型
    print(f"載入原始模型： {original_model_path}")
    orig_model = onnx.load_model(original_model_path)
    new_model = onnx.load_model(original_model_path) # 建立一個副本進行修改

    # 找到原始輸入的 TensorProto 資訊
    original_input_info = None
    for input_proto in orig_model.graph.input:
        if input_proto.name == input_to_replace:
            original_input_info = input_proto
            break

    if original_input_info is None:
        raise ValueError(f"在模型中找不到名為 '{input_to_replace}' 的輸入。請檢查輸入名稱是否正確。")

    # 檢查原始輸入是否為 uint8
    if original_input_info.type.tensor_type.elem_type != TensorProto.UINT8:
        print(f"警告：原始輸入 '{input_to_replace}' 的資料類型不是 UINT8，而是 {TensorProto.DataType.Name(original_input_info.type.tensor_type.elem_type)}")
        print("程式碼仍會插入 Cast 節點，但請確保這是您想要的行為。")

    # 定義新的輸入名稱和類型
    # 我們將把原始輸入重新命名為 'old_input_uint8'，並創建一個新的 float32 輸入
    new_input_name = f"{input_to_replace}_float32"
    
    # 1. 修改圖的輸入
    # 清空現有的輸入，然後添加新的 float32 輸入，以及將原始 uint8 輸入重新命名
    del new_model.graph.input[:]
    
    # 添加新的 float32 輸入
    # 形狀應該與原始輸入相同，但這裡我們使用原始輸入的形狀
    new_model.graph.input.append(
        helper.make_tensor_value_info(
            new_input_name,
            TensorProto.FLOAT, # 新的資料類型為 FLOAT32
            # 複製原始輸入的形狀 (假設是動態形狀，可以直接複製 dim_param)
            [d.dim_value if d.HasField('dim_value') else d.dim_param for d in original_input_info.type.tensor_type.shape.dim]
        )
    )

    # 創建一個新的 Cast 節點
    # 這個節點將新的 float32 輸入轉換回原始模型節點所需的 uint8 類型
    # 確保輸出名稱是原始模型的輸入名稱，這樣就可以無縫連接到原始模型的圖
    cast_node = helper.make_node(
        op_type="Cast",
        inputs=[new_input_name],  # Cast 節點的輸入是新的 float32 輸入
        outputs=[input_to_replace], # Cast 節點的輸出是原始模型節點期望的輸入名稱
        name="Cast_uint8_to_float32", # 節點名稱
        to=TensorProto.UINT8 # 轉換目標類型為 UINT8
    )

    # 2. 修改圖的節點
    # 清空所有現有的節點，然後添加 Cast 節點，再添加原始模型的節點
    # 確保 Cast 節點在原始節點之前，以保持拓撲順序
    del new_model.graph.node[:]
    new_model.graph.node.append(cast_node)
    new_model.graph.node.extend(orig_model.graph.node)

    # 3. 檢查和儲存修改後的模型
    print("檢查修改後的模型...")
    onnx.checker.check_model(new_model)
    print("模型檢查成功！")

    print(f"儲存修改後的模型到： {output_model_path}")
    onnx.save_model(new_model, output_model_path)
    print("模型修改並儲存完成。")
    print(f"現在，您的模型 '{output_model_path}' 應該接受 Float32 輸入。")

except Exception as e:
    print(f"處理模型時發生錯誤： {e}")

