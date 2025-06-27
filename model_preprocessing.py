import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

# 定義您的原始模型檔案名稱
# 重要：請將此替換為您從 tf2onnx 轉換出來的 ONNX 檔案路徑。
# 例如：original_model_path = "new_model_v1/model_v1_for_trt.onnx"
original_model_path = "new_model_fpnlite/model.onnx" # <--- 請務必更新為您的實際檔案路徑！
output_model_path = "new_model_fpnlite/model_float32.onnx" # 修正後輸出的檔案名稱

# 根據您在 Netron 中找到的 initializer 值，將 K_VALUE 設定為 100。
# 如果您的模型 K 值不同，請務必修改此處。
FIXED_K_VALUE = 100 # <--- 請確認此值是否正確！

try:
    print(f"載入原始模型： {original_model_path}")
    orig_model = onnx.load_model(original_model_path)
    
    # --- 第 1 部分：修改模型輸入為 FLOAT32 ---
    # 假設您的主要輸入名稱是 'input_tensor:0'。請檢查 Netron 確認。
    input_to_modify = 'input_tensor:0' 
    original_input_info = None
    # 尋找原始模型的輸入資訊
    for input_proto in orig_model.graph.input:
        if input_proto.name == input_to_modify:
            original_input_info = input_proto
            break

    if original_input_info is None:
        raise ValueError(f"在模型中找不到名為 '{input_to_modify}' 的輸入。請檢查輸入名稱是否正確。")

    # 檢查原始輸入是否為 UINT8，並給出提示
    if original_input_info.type.tensor_type.elem_type != TensorProto.UINT8:
        print(f"警告：原始輸入 '{input_to_modify}' 的資料類型不是 UINT8，而是 {TensorProto.DataType.Name(original_input_info.type.tensor_type.elem_type)}")
        print("將繼續轉換為 FLOAT32。")

    # 清空原始模型的輸入列表，並添加新的 FLOAT32 輸入
    del orig_model.graph.input[:] 
    orig_model.graph.input.append(
        helper.make_tensor_value_info(
            input_to_modify, # 使用原始輸入名稱
            TensorProto.FLOAT, # 新的資料類型為 FLOAT32
            # 複製原始輸入的形狀 (處理動態維度，如 'N' 或 'unk_xxx')
            [d.dim_value if d.HasField('dim_value') else d.dim_param for d in original_input_info.type.tensor_type.shape.dim]
        )
    )
    print(f"模型輸入 '{input_to_modify}' 已成功轉換為 FLOAT32。")

    # 為了方便後續操作，複製原始模型的節點、初始值和值資訊。
    # 注意：orig_model.graph.input 已經在上面被修改了。
    nodes = []
    initializers = list(orig_model.graph.initializer) 
    value_info = list(orig_model.graph.value_info) 

    # --- 第 2 部分：處理 TopK 節點的 K 值為 Constant ---
    for node in orig_model.graph.node: # 遍歷原始模型的節點
        if node.op_type == "TopK":
            print(f"找到 TopK 節點: {node.name}")
            if len(node.input) > 1: # TopK 應該至少有兩個輸入 (data, k)
                k_input_name = node.input[1]
                
                # 檢查 K 值是否已經是模型中的一個初始值 (Constant)。
                # tf2onnx --target tensorrt 可能已經處理了這個問題。
                k_is_initializer = False
                for init in initializers:
                    if init.name == k_input_name:
                        k_is_initializer = True
                        break
                
                if not k_is_initializer: # 如果 K 不是初始值，則我們插入 Constant 節點來替換它
                    print(f"  TopK 的 K 輸入名稱: {k_input_name} (非 Constant，正在轉換)")
                    k_constant_name = f"{node.name}_K_value_const" # 為新的 Constant 節點命名
                    k_tensor = helper.make_tensor(
                        name=k_constant_name,
                        data_type=TensorProto.INT64, # K 值通常是 INT64 類型
                        dims=[], # 標量 (scalar) 形狀為空列表
                        vals=[FIXED_K_VALUE] # 設定為我們定義的固定 K 值
                    )
                    # 避免重複添加相同的初始值
                    if not any(init.name == k_constant_name for init in initializers):
                        initializers.append(k_tensor) 
                    
                    # 創建一個新的 TopK 節點，其第二個輸入指向我們剛剛創建的 Constant 節點
                    new_topk_node = helper.make_node(
                        op_type="TopK",
                        inputs=[node.input[0], k_constant_name], # 第一個輸入不變，第二個輸入替換為 Constant
                        outputs=node.output, # 輸出保持不變
                        name=node.name, # 節點名稱保持不變
                        domain=node.domain # 節點網域保持不變
                    )
                    # 複製所有原始屬性 (例如 'sorted' 屬性)
                    for attr in node.attribute:
                        new_topk_node.attribute.append(attr)
                    nodes.append(new_topk_node) # 將新創建的節點添加到列表中
                    print(f"  TopK 節點 '{node.name}' 的 K 輸入已替換為 Constant 值 {FIXED_K_VALUE}")
                else: # K 已經是初始值，直接使用原始節點
                    print(f"  TopK 的 K 輸入名稱: {k_input_name} (已是 Constant，無需修改)")
                    nodes.append(node)
            else:
                print(f"警告: TopK 節點 '{node.name}' 只有一個輸入，可能不是我們預期的情況。將保持原樣。")
                nodes.append(node) # 將其他不符合條件的 TopK 節點原樣添加
        else:
            nodes.append(node) # 將非 TopK 節點原樣添加

    # 重新構建 ONNX 圖，使用我們修改後的節點列表、初始值和輸入列表
    new_graph = helper.make_graph(
        nodes=nodes,
        name=orig_model.graph.name,
        inputs=orig_model.graph.input, # 使用已經修改為 FLOAT32 的輸入
        outputs=orig_model.graph.output,
        initializer=initializers,
        value_info=value_info
    )

    # 創建新的 ONNX 模型物件
    new_model = helper.make_model(new_graph, producer_name=orig_model.producer_name)
    
    # --- 第 3 部分：正確處理 opset_import ---
    # 首先，清空新模型可能已經存在的 opset_import 列表。
    del new_model.opset_import[:] 
    # 然後，從原始模型中複製所有 opset_import 資訊。
    # 這樣可以保留 tf2onnx 添加的任何特定網域的 Opsets。
    for opset in orig_model.opset_import:
        new_model.opset_import.append(opset)
    
    # 確保 'ai.onnx' 網域的 Opset 版本是 11。如果已經存在則更新，否則添加。
    found_ai_onnx_opset_11 = False
    for opset in new_model.opset_import:
        if opset.domain == "ai.onnx":
            opset.version = 11 # 強制設定為 11
            found_ai_onnx_opset_11 = True
            break
    
    if not found_ai_onnx_opset_11:
        new_model.opset_import.append(helper.make_opsetid("ai.onnx", 11))

    # 檢查和儲存修改後的模型
    print("檢查修改後的模型...")
    onnx.checker.check_model(new_model) # 執行 ONNX 檢查器以驗證模型有效性
    print("模型檢查成功！")

    print(f"儲存修改後的模型到： {output_model_path}")
    onnx.save_model(new_model, output_model_path) # 將修改後的模型儲存到檔案
    print("模型修改並儲存完成。")
    print(f"現在，您的模型 '{output_model_path}' 的輸入是 Float32，並且 TopK 節點 K 值已處理。")
    print("\n提醒：在 TensorRT 推論前，請確保您的影像資料已轉換為 Float32 並進行正規化 (例如除以 255.0)。")

except Exception as e:
    print(f"處理模型時發生錯誤： {e}")

