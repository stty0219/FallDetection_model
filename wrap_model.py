import tensorflow as tf
import os
# 引入 convert_variables_to_constants_v2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# --- 配置路徑 ---
original_saved_model_path = r'C:\Users\a0931\TensorFlow\new_model_v1\saved_model'
new_saved_model_path = r'C:\Users\a0931\TensorFlow\new_model_v1\saved_model_with_preprocessing'

# --- 載入原始 SavedModel ---
print(f"載入原始 SavedModel: {original_saved_model_path}")
loaded_model = tf.saved_model.load(original_saved_model_path)

# 獲取原始模型的推理簽名
infer_original = loaded_model.signatures['serving_default']
print(f"原始模型輸入簽名: {infer_original.structured_input_signature}")

# --- 關鍵修改：獲取一個可以被凍結的 ConcreteFunction ---
# 我們不能直接對 infer_original 調用 get_concrete_function。
# 相反，我們需要構造一個新的 tf.function，讓它來調用 infer_original，
# 然後再從這個新的 tf.function 中獲取 ConcreteFunction。
# 這樣 TensorFlow 才能正確地建立圖。

# 1. 定義一個臨時的 tf.function，它簡單地將輸入傳遞給原始模型的推理函數
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 300, 300, 3], dtype=tf.uint8, name='temp_input_tensor')
])
def _temp_wrapper_for_freezing(temp_input):
    # 這裡的 input_tensor 是原始模型期待的輸入名稱
    return infer_original(input_tensor=temp_input)

# 2. 從這個臨時的 tf.function 中獲取一個具體函數
#    這一步會觸發 TensorFlow 構建圖，以便我們可以進行凍結。
#    如果你的原始模型輸入簽名是 ((), {'input_tensor': TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8, name='input_tensor')})
#    那麼這裡我們用一個固定的 shape 來獲取具體函數
concrete_function = _temp_wrapper_for_freezing.get_concrete_function(
    tf.TensorSpec(shape=[1, 300, 300, 3], dtype=tf.uint8, name='temp_input_tensor')
)

# 3. 將具體函數中的所有變量轉換為常數
frozen_func = convert_variables_to_constants_v2(concrete_function)

# --- 定義一個新的 tf.Module 來包裹凍結後的模型和預處理邏輯 ---
class PreprocessingModel(tf.Module):
    def __init__(self, frozen_inference_fn):
        super().__init__()
        self.frozen_inference_fn = frozen_inference_fn
        # 從 frozen_func 的 outputs 獲取輸出名稱
        # 這通常是一個元組或列表，每個元素是 Tensor。
        # 對於 Object Detection API，我們知道輸出名稱
        self.output_names = [
            'detection_anchor_indices', 'detection_boxes', 'detection_classes',
            'detection_multiclass_scores', 'detection_scores', 'num_detections',
            'raw_detection_boxes', 'raw_detection_scores'
        ]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 300, 300, 3], dtype=tf.uint8, name='image_tensor_uint8') 
    ])
    def __call__(self, image_tensor_uint8):
        # 1. 將 uint8 轉換為 float32
        image_tensor_float32 = tf.cast(image_tensor_uint8, dtype=tf.float32)

        # 2. 執行歸一化
        normalized_image_tensor = (image_tensor_float32 / 127.5) - 1.0 
        
        # 3. 將預處理後的 float32 張量，傳遞給凍結後的模型函數
        #    frozen_func 期望的是位置參數，這裡我們傳遞一個張量
        #    並且需要注意，frozen_func 的輸入是 'input_tensor' (原始模型的輸入)，
        #    但我們這裡傳遞的其實是歸一化後的張量
        #    frozen_func 的 input_names 列表中的第一個應該是我們想要的。
        
        # 由於 frozen_func 的輸入可能會有不同的名稱，我們直接將 normalized_image_tensor 傳入
        # 並假設它是 frozen_func 的第一個且唯一的圖片輸入
        output_tensors = self.frozen_inference_fn(normalized_image_tensor)
        
        # frozen_func 返回的是一個元組的張量，將其轉換為字典
        # 確保 output_names 的順序與 frozen_func 返回的張量順序一致
        output_dict = {name: tensor for name, tensor in zip(self.output_names, output_tensors)}
        
        return output_dict

# 實例化我們的包裝模型
wrapped_model = PreprocessingModel(frozen_func)

# --- 保存新的 SavedModel ---
print(f"保存新的 SavedModel 到: {new_saved_model_path}")
tf.saved_model.save(
    wrapped_model,
    new_saved_model_path,
    signatures={
        'serving_default': wrapped_model.__call__.get_concrete_function(
            tf.TensorSpec(shape=[1, 300, 300, 3], dtype=tf.uint8, name='image_tensor_uint8')
        )
    }
)
print("新的 SavedModel (包含預處理) 已成功創建。")