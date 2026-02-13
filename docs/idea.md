根據你整理的可微動態規劃與序列對齊方法分類表，我提出兩個具有理論深度且可操作的 toy experiment 方向：

***

## Experiment 1: Temperature-Induced Grokking in Soft Alignment

### 核心假設
Soft-DTW 和 differentiable Smith-Waterman 中的溫度參數 \(\tau\) 不僅控制「soft vs. hard」，還可能觸發類似 **grokking** 的相變現象——在某個臨界溫度下，模型從記憶局部捷徑突然切換到學習真實生物對齊規則。

### 實驗設計
- 固定簡單任務：合成的蛋白質家族序列（已知真實對齊）
- 變量：\(\tau \in [0.01, 10]\)，訓練 epoch 數
- 觀測指標：
  1. **Training loss** vs. **true alignment accuracy**（用 column score 或 sum-of-pairs 衡量）
  2. 對齊路徑的 **entropy** 隨訓練的演化
  3. Scoring matrix 的 **rank** 或 **effective dimension**

### 預期理論貢獻
如果發現存在「溫度臨界區間」使得模型從過擬合轉向泛化（類似 grokking 的 generalization phase transition），可以：
- 建立 **\(\tau\)-dependent capacity theory**：低溫過早硬化，高溫過度模糊，中間存在最優學習regime
- 連結到 **statistical mechanics of alignment**：將對齊分佈視為 Boltzmann 分佈，用自由能解釋相變
- 提供 **principled temperature scheduling** 策略，不再靠經驗調參

***

## Experiment 2: Adversarial Consistency Regularization for Differentiable Alignment

### 核心假設
當 alignment layer 與下游黑盒目標（如 AlphaFold confidence）聯合訓練時，模型可能學到「提高下游指標但違反生物一致性」的對抗性對齊。這類似對抗樣本，但發生在**結構化輸出空間**（alignment paths）。

### 實驗設計
設計 **minimal oracle**：
- 給定 pair/MSA，已知真實結構距離矩陣 \(D_{\text{true}}\)
- Baseline：用 LAM style 端到端訓練（只優化下游 contact prediction loss）
- Intervention：加入 **consistency regularizer**：
  \[
  \mathcal{L}_{\text{reg}} = \text{KL}(P_{\text{align}}(\tau_1) \| P_{\text{align}}(\tau_2))
  \]
  強制不同溫度下的對齊分佈保持一致

- 測量：
  1. **Alignment deviation** from biological prior（用 BLOSUM/PAM 計算偏離）
  2. **Downstream performance** 是否仍維持
  3. 是否出現「temperature-dependent collapse」（某些溫度下對齊完全崩壞）

### 預期理論貢獻
- 證明 **unconstrained differentiable alignment 的 non-identifiability**：存在無限多對齊可達成相同下游損失
- 提出 **biologically-informed inductive bias**：將對齊視為受物理/演化約束的 structured prediction
- 發展 **multi-temperature ensemble theory**：證明 temperature averaging 等價於對齊空間的某種 regularization
- 可能連結到 **causal alignment**：某些對齊路徑是「因果必要的」（結構決定），某些是「偶然的」（可替換）

***

## 兩個實驗的共通價值

1. **理論可證明性**：都可能導出閉式結果（溫度相變的臨界指數、正則化的 PAC bound）
2. **實作簡潔**：只需修改現有 Soft-DTW/LAM 框架，計算成本低
3. **廣泛影響**：結論可遷移到所有「可微 DP + 下游任務」場景（不限生物序列）

這兩個方向都觸及你表格中未被明確討論的理論盲點——**soft alignment 的學習動力學**與**多目標訓練的結構化對抗性**。