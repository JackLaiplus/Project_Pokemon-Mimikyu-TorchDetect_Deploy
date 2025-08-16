# Project_Pokemon-Mimikyu-TorchDetect

<p align="center">
  <img src="assets/mimikyu.png" alt="Mimikyu Banner" width="90%">
</p>

專案簡介，`Project_Pokemon-Mimikyu-TorchDetect` 是一個基於 **TensorFlow-Keras** 的圖像分類專案，專門用來辨識寶可夢 **Mimikyu** 與 **Pikachu**。Mimikyu（日語：ミミッキュ）是一種會偽裝成 Pikachu 的寶可夢，原因在於為了獲得更多朋友。由於外觀相似，對模型來說辨識具有挑戰性。本專案目的是透過深度學習模型 **EfficientNetB0**，準確區分這兩種寶可夢。

### 🎯 專案目標
1. **建立高準確度分類模型**  
   使用 EfficientNetB0 作為骨幹網路（Backbone），在有限的資料下達到最佳分類效果。
2. **解決外觀相似的分類挑戰**  
   Mimikyu 與 Pikachu 在外觀上有高度相似性，模型需能正確分辨兩者。
3. **提供可重現的訓練流程**  
   包含資料增強、模型訓練、微調（Fine-tuning）與最佳權重儲存。
4. **支援單張與批量推論**  
   方便部署與展示模型推論結果。
5. **便於模型部署**  
   支援輸出 `.keras` 與 `.h5` 格式，方便在不同環境載入與使用。

### 📊 資料集說明
本專案的資料集格式與 [Kaggle: Pokémon Gen 1 - 151 Classes Classification](https://www.kaggle.com/datasets/hongdcs/pokemon-gen1-151-classes-classification) 類似，但僅保留 **Mimikyu** 與 **Pikachu** 兩類，並依照訓練需求進行資料切分。

### 🛠️ 使用技術
- Python 3.8+
- NumPy
- Matplotlib
- Keras
- Pillow
- Jupyter Notebook

### 📊 批量推論結果

以下為模型在測試集（Test set）上隨機抽樣 12 張圖片進行批量推論的結果：
<p align="center">
  <img src="assets/batch_prediction_test.png" alt="Batch Inference Results" width="90%">
</p>


