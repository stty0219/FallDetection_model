import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

# 定義您的原始模型檔案名稱
original_model_path = "new_model_v1/model_v1_opset11_float32_noCast.onnx" # 請使用您上次成功生成的檔案
output_model_path = "new_model_v1/model_v1_fixed_topk_final.onnx" # 新的輸出檔案名

# 根據您在 Netron 中找到的 initializer 值，將 K_VALUE 設定為 100
FIXED_K_VALUE = 100 # <--- 已確認為 100

try:
    print(f"載入原始模型： {original_model_path}")
    orig_model = onnx.load_model(original_model_path)
    
    # 創建一個新的圖來構建修改後的模型
    nodes = []
    initializers = list(orig_model.graph.initializer) # 複製原始的初始值
    value_info = list(orig_model.graph.value_info) # 複製原始的 value_info

    # 遍歷原始圖中的所有節點
    for node in orig_model.graph.node:
        if node.op_type == "TopK":
            print(f"找到 TopK 節點: {node.name}")
            # 檢查 TopK 節點的第二個輸入 (K值)
            if len(node.input) > 1:
                k_input_name = node.input[1]
                print(f"  TopK 的 K 輸入名稱: {k_input_name}")

                # 創建一個新的 Constant 節點來表示 K 值
                # K 值必須是一個 INT64 標量張量 (shape [])
                k_constant_name = f"{node.name}_K_value_const"
                k_tensor = helper.make_tensor(
                    name=k_constant_name,
                    data_type=TensorProto.INT64, # K 值通常是 INT64
                    dims=[], # 標量 (scalar)
                    vals=[FIXED_K_VALUE]
                )
                # 檢查這個 initializer 是否已經存在，避免重複添加
                if not any(init.name == k_constant_name for init in initializers):
                    initializers.append(k_tensor) # 將 K 值添加到初始值列表
                
                # 創建一個新的 TopK 節點，將其第二個輸入替換為我們的新 Constant 節點
                new_topk_node = helper.make_node(
                    op_type="TopK",
                    inputs=[node.input[0], k_constant_name], # 第一個輸入不變，第二個輸入替換為 Constant
                    outputs=node.output,
                    name=node.name,
                    domain=node.domain # 保持原始 domain
                )
                # 複製所有原始屬性，包括 'sorted'
                for attr in node.attribute:
                    new_topk_node.attribute.append(attr)
                
                nodes.append(new_topk_node)
                print(f"  TopK 節點 '{node.name}' 的 K 輸入已替換為 Constant 值 {FIXED_K_VALUE}")
            else:
                print(f"警告: TopK 節點 '{node.name}' 只有一個輸入，可能不是我們預期的情況。")
                nodes.append(node) # 保持原樣
        else:
            nodes.append(node) # 將其他節點原樣添加

    # 重新構建圖，使用修改後的節點列表和初始值列表
    # 輸入和輸出保持原始模型的
    new_graph = helper.make_graph(
        nodes=nodes,
        name=orig_model.graph.name,
        inputs=orig_model.graph.input,
        outputs=orig_model.graph.output,
        initializer=initializers,
        value_info=value_info
    )

    new_model = helper.make_model(new_graph, producer_name=orig_model.producer_name)
    
    # 修正：正確處理 opset_import
    # 首先，複製原始模型的所有 opset_import
    del new_model.opset_import[:] # <--- 修正：使用 del 來清空列表
    for opset in orig_model.opset_import:
        new_model.opset_import.append(opset)
    
    # 確保 ai.onnx 網域的 Opset 版本是 11。如果已經存在且版本不同，則更新它。
    # 如果不存在，則添加它。
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
    onnx.checker.check_model(new_model)
    print("模型檢查成功！")

    print(f"儲存修改後的模型到： {output_model_path}")
    onnx.save_model(new_model, output_model_path)
    print("模型修改並儲存完成。")
    print(f"現在，您的模型 '{output_model_path}' 的 TopK 節點 K 值已變為 Constant。")

except Exception as e:
    print(f"處理模型時發生錯誤： {e}")
