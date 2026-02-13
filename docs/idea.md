根據您整理的可微對齊 taxonomy，我提出兩個能引導 strong theoretical contribution 的 insightful toy experiments：

## Experiment 1: Alignment Collapse under Multi-Objective Optimization

### 核心假設
當同時優化 soft alignment loss（如 Soft-DTW）和下游任務目標（如分類/預測）時，系統可能學到「局部最優但生物學不合理」的對齊策略。這類似您表中提到的「學到對抗性 alignment」問題，但我們關注的是**多目標間的 Pareto frontier 是否存在結構性斷層**。

### 實驗設計
- 使用合成序列對（ground truth alignment 已知）
- 設計三種下游目標：
  1. 需要精確對齊的任務（如 position-specific prediction）
  2. 只需粗略對齊的任務（如 global similarity classification）
  3. 與對齊無關但與序列特徵相關的任務
- 追蹤訓練過程中：
  - Soft alignment entropy（衡量對齊的「軟度」）
  - Ground truth alignment recovery rate
  - 下游任務性能
  - **關鍵**：計算 alignment gradient 與 task gradient 的 cosine similarity

### 預期 Insight
如果發現某些溫度參數或任務組合下，alignment 會突然「collapse」到高熵但任務性能不降的狀態，這暗示：
- 需要新的正則化方法來維持對齊的結構性
- 可能需要 curriculum learning 策略（先學對齊再學任務）
- 或者發展**動態溫度調節機制**（根據梯度衝突自適應調整）

***

## Experiment 2: Smoothness-Hardness Trade-off in DP Path Probability Concentration

### 核心假設
Differentiable DP 框架中，平滑算子（如 logsumexp）的溫度參數控制了路徑機率分佈的集中度。但**不同 DP 問題類別對「最優平滑度」的需求可能遵循不同的 scaling law**。

### 實驗設計
- 構造三類 DP 問題：
  1. **單峰問題**：最優路徑明確唯一（如簡單 alignment）
  2. **多峰問題**：存在多條近似最優路徑（如有重複 motif 的序列）
  3. **平坦問題**：大量路徑的 cost 相近（如隨機序列對齊）
- 對每類問題，掃描溫度參數 \(\tau \in [0.01, 10]\)，測量：
  - Path entropy \(H = -\sum_{\text{path}} p \log p\)
  - Gradient signal-to-noise ratio（SNR）
  - 收斂速度與最終性能
  - **關鍵**：計算 effective path count \(\exp(H)\) 與最優解距離的關係

### 預期 Insight
如果發現：
- 單峰問題在 \(\tau \to 0\) 時梯度 SNR 最高
- 多峰問題需要中等 \(\tau\) 來平衡 exploration
- 平坦問題在任何 \(\tau\) 下都難以學習

則可發展**problem-aware temperature scheduling**：
- 用 meta-learning 自動診斷問題類別
- 動態調整 \(\tau\)（類似 simulated annealing 但基於梯度統計）
- 或者設計 multi-temperature ensemble（不同 head 用不同 \(\tau\)，最後 attention fusion）

***

## 為何這兩個實驗有理論價值

1. **非必然性**：現有文獻多假設「平滑就能優化」，但沒有系統研究**何時平滑會傷害結構學習**
2. **可泛化**：發現適用於整個 differentiable DP 家族，不限於特定 alignment 方法
3. **可操作**：insights 直接導向新方法（adaptive temperature、gradient conflict resolution）
4. **連接理論與實踐**：將 optimization landscape 幾何（多目標 Pareto、loss landscape curvature）與生物序列特性（motif structure、conservation pattern）關聯起來

***

這兩個方向是否符合您研究的脈絡？還是您想探索更偏向某個特定應用場景（如 FGVC 中的 part alignment）的 toy experiment?