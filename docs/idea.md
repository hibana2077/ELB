## Research Proposal：Energy Landscape-Based Subclass via Morse-Theoretic Stratification (ELB)

### 0) 問題背景與動機

現有的 *subclass discovery* 常把 subclass 當成「幾何距離上的 cluster」：依賴 Euclidean/embedding 距離、或 mixture 假設（GMM / k-means / spectral）。但在高維資料（影像、醫療、語音、表示學習 embedding）中，**距離不一定對應語意結構**，而且 cluster 邊界容易被噪聲或表示扭曲破壞。

本研究提出：**不以 cluster 定義 subclass，而以樣本密度所誘導的能量地形之拓撲穩定 basin 定義 subclass**。Subclass 變成一個「動力系統的吸引域」與「拓撲物件」，核心依據是 Morse theory / gradient flow / critical point theory 的穩定性。

---

## 1) 研究目標 (Research Objectives)

1. **形式化 subclass 的新定義**：給定平滑密度 (p(x))，以能量 (E(x)=-\log p(x)) 的梯度流之 basin-of-attraction 定義 subclass（而非距離群集）。
2. **建立理論保證**：在合理條件（(E) 為 Morse / Morse–Smale）下，證明 subclass（basin 分割）對小密度擾動具有拓撲穩定性（critical point index 不變、stable manifold 分割同胚/微分同胚）。
3. **提出可實作方法**：以 score-based model / normalizing flow 等估計 (p(x)) 或 (\nabla \log p(x))，用 ODE/gradient flow 指派 basin，並以 Hessian signature 分析 saddle 結構與 subclass 邊界。
4. **驗證「不是 cluster」的效益**：在 toy 與真實資料上，展示 ELB 對於非凸形狀、低密度 ridge、長尾子族群（rare subclass）更穩定，並可改善下游分類/偵測/解釋性。

---

## 2) 理論洞見 (Theoretical Insights)

* **Subclass 是拓撲不變的吸引域分解**：資料空間（或 embedding manifold）被 (E(x)) 的梯度流分割成 stable manifolds；每個 local minimum 的 stable manifold 即一個 subclass。
* **邊界由 saddle 的 unstable manifold 決定**：subclass 邊界不是「距離等分線」，而是由 index-1 saddle 的 separatrix（分離流形）構成；這提供可解釋的「為何這些點被分開」。
* **穩定性來自 Morse 理論**：只要 (E) 維持 Morse（臨界點非退化）且流場滿足 Morse–Smale（stable/unstable manifold 橫截），則小擾動不改變拓撲分解的型態 → subclass 定義具 robust 的理論基礎。

---

## 3) 創新點 (Innovation)

1. **Subclass 定義從幾何 → 動力系統 / 拓撲**：不以距離或 mixture 假設，而以 (E=-\log p) 的拓撲穩定 basin 定義 subclass。
2. **把 mean-shift / density clustering 從 heuristic 提升為可證明的 Morse 架構**：同樣是沿密度梯度走向 mode，但本研究顯式使用 critical point index、Morse 分解、以及穩定性條件。
3. **saddle-aware 的 subclass 邊界刻畫**：用 Hessian signature（特徵值符號）辨識 minima/saddle，將 subclass 邊界與 saddle 結構連結，提供「可驗證、可視化、可推導」的決策結構。
4. **可做 multi-scale / persistent subclass**：透過密度平滑尺度（KDE bandwidth 或 score model 的 noise level），得到隨尺度變化的 basin 合併/分裂，形成 subclass 的 persistence 分析（更貼近真實資料的層級結構）。

---

## 4) 研究貢獻 (Contributions)

* **C1（定義）**：提出 ELB subclass = 能量地形 basin 的形式化定義。
* **C2（理論）**：給出 subclass 拓撲穩定性的充分條件與定理（Morse / Morse–Smale + 小 (C^2) 擾動）。
* **C3（演算法）**：一套可在高維中運作的 basin 指派、critical point 搜尋、Hessian index 估計與邊界重建流程。
* **C4（實證）**：在合成與真實資料上，展示 ELB 對於非凸、低密度 ridge、rare subclass 的優勢與穩健性，並量化 stability。

---

## 5) 理論貢獻 (Theoretical Contributions)

### 5.1 基本定義（Subclass 作為 stable manifold）

給定平滑密度 (p(x))（定義在 (\mathbb{R}^d) 或流形 (\mathcal{M}) 上），能量
[
E(x) = -\log p(x).
]
考慮梯度流（gradient flow）
[
\frac{dx}{dt} = -\nabla E(x) = \nabla \log p(x).
]
令 (\phi_t(x)) 表示流的解（flow map）。若 (m_k) 是 (E) 的 local minimum（亦即 (p) 的 mode），定義 basin / subclass：
[
\mathrm{Subclass}*k ;=; W^s(m_k) ;=; {x:\lim*{t\to\infty}\phi_t(x)=m_k}.
]
此即 Morse theory 中的 **stable manifold**。整體空間（除 measure-zero 的分界集合）被這些 stable manifolds 分割。

### 5.2 臨界點分類（Hessian signature）

臨界點 (x^\star) 滿足 (\nabla E(x^\star)=0)。其 Morse index 定義為 Hessian 的負特徵值個數：
[
\mathrm{index}(x^\star)=#{\lambda_i(\nabla^2E(x^\star))<0}.
]

* index (0)：local minimum（mode / subclass 代表）
* index (1)：一階 saddle（典型 subclass 分界的「閘口」）
* index (>1)：更高階 saddle（更複雜分界結構）

### 5.3 Subclass 穩定性（核心定理主張）

> **定理 A（Morse 臨界點穩定性，概念版）**
> 若 (E) 為 Morse 函數（所有臨界點非退化：(\det\nabla^2E(x^\star)\neq 0)），且 (\tilde{E}=E+\Delta E) 與 (E) 在 (C^2) 範數下足夠接近（(|\Delta E|_{C^2}<\varepsilon)），則：
>
> 1. (\tilde{E}) 的臨界點與 (E) 的臨界點存在一一對應（在局部鄰域內），
> 2. 對應臨界點的 Morse index 相同（minima/saddle 型態不變）。

> **定理 B（Morse–Smale 分解穩定性，概念版）**
> 若梯度流 (\dot{x}=-\nabla E(x)) 為 Morse–Smale（stable/unstable manifolds 橫截），則對足夠小的 (C^1)（或 (C^2)）擾動 (\tilde{E})，存在同胚/微分同胚 (h) 使得
> [
> h(W^s_E(m_k)) = W^s_{\tilde{E}}(\tilde{m}_k),
> ]
> 因此 subclass 的 basin 分割在拓撲意義下保持不變（除邊界附近的可控擾動）。

**意義**：只要密度估計或資料擾動不造成臨界點「退化或湮滅/生成」（即不跨越臨界事件），ELB subclass 就具有可證明的穩健性。

---

## 6) 方法論 (Methodology)

### 6.1 密度 / score 的學習

我們有兩條路徑（可比較）：

**(M1) Score-based model（推薦，避免顯式密度）**
學 (;s_\theta(x)\approx\nabla \log p(x))。
優點：高維可行、直接就是 flow 方向；可用多噪聲層級做 multi-scale subclass。

**(M2) Normalizing flow / energy-based density**
學 (\log p_\theta(x)) 進而得到 (E_\theta(x)=-\log p_\theta(x))、(\nabla E_\theta)、(\nabla^2 E_\theta)。
優點：能量與密度可評估；缺點：某些資料更難訓練。

### 6.2 Basin 指派（Subclass assignment）

對每個樣本 (x)，做 ODE 積分（或離散梯度下降）：
[
x_{t+1}=x_t-\eta\nabla E(x_t) \quad(\text{等價於 } x_{t+1}=x_t+\eta,\nabla\log p(x_t)).
]
收斂到的 minimum (m_k) 即 subclass label。
為避免不同 run 收斂到近似相同 minimum，可在 minima 空間做 merge（例如以能量差、距離與 Hessian 近似相等作同一 mode）。

### 6.3 Critical point 偵測與 Hessian index

* **臨界點搜尋**：從多個起點做 gradient flow/局部 Newton（在 score=0 的條件下），收集候選 minima/saddle。
* **Hessian index 估計**：高維下用 Lanczos / power iteration / Hutchinson trace 近似取得負特徵值數，判斷 index。
* **邊界刻畫**：聚焦 index-1 saddle，沿其 unstable manifold 追蹤 separatrix，重建 basin 邊界的幾何結構（可在低維投影空間或局部 chart 近似）。

### 6.4 Multi-scale / Persistent subclass（選配但很強）

定義一族平滑後能量 (E_\sigma)（例如 KDE bandwidth、或 score model 的 noise level）。隨 (\sigma) 變化，basin 會合併/分裂。
輸出：subclass 的層級樹（merge tree）或 persistence summary，讓 subclass 不只是一個固定 partition，而是尺度可控的拓撲結構。

---

## 7) 數學理論推演與證明（Proof Plan，偏「可投稿」的寫法）

本研究預計在論文中給出以下可形式化的推導骨架（依投稿 venue 可調嚴謹度）：

1. **從密度到 Morse 函數的條件**

   * 假設 (p(x)\in C^2)、且在考慮區域內 (p(x)>0) → (E=-\log p\in C^2)。
   * 假設臨界點滿足非退化（(\nabla E=0\Rightarrow \det\nabla^2E\neq 0)）→ (E) 為 Morse。

2. **臨界點與 index 的穩定性（定理 A）**

   * 以隱函數定理（implicit function theorem）處理 (\nabla \tilde{E}(x)=0) 在臨界點附近的解存在與唯一性。
   * 以 Hessian 特徵值連續性與 Weyl 不等式控制特徵值符號不翻轉 → index 保持。

3. **basin 分割的穩定性（定理 B）**

   * 引入 Morse–Smale 條件確保 stable/unstable manifolds 橫截，並使用動力系統的結構穩定性（structural stability）結論：小擾動下流的拓撲共軛（topological conjugacy）。
   * 推得 stable manifolds 的分割在同胚下對應 → subclass 穩定。

4. **可計算性與誤差傳遞（從估計的 score 到 basin label）**

   * 若 (s_\theta(x)) 與真實 (\nabla \log p(x)) 在區域內一致逼近（(|s_\theta-\nabla \log p|_\infty\le \delta)），可推導離散 flow 的終點偏差與 basin label 翻轉的條件：翻轉主要發生在靠近 separatrix 的薄層，並可用 margin（到邊界的最小流形距離）給出上界。

---

## 8) 預計使用 Dataset

（目的：同時驗證「可視化 toy」與「高維真實資料」）

1. **Toy / 合成資料（必做）**

   * 兩個高斯 + 中間低密度 ridge / 薄橋（你描述的 setting）
   * 非凸月牙（two-moons）、環狀（annulus）、多模態且 saddle 控制邊界的例子
     評估：ELB basin 是否與真實生成機制一致、k-means 是否錯分、邊界是否落在低密度區。

2. **影像資料（建議）**

   * MNIST / Fashion-MNIST：做 subclass（例如同一類數字內的筆劃風格 basin）
   * CIFAR-10：在 class-conditional density 上找 subclass（同一類物體的姿態/背景 basin）
   * 若你偏醫學：MedMNIST 系列做 class 內亞型（domain shift + rare patterns）

3. **表格 / 表徵資料（選配）**

   * UCI 或 embedding dataset（例如 sentence embeddings、病歷特徵向量）
     重點：展示距離型分群在 embedding distortion 下不穩，而 ELB 仍能靠低密度邊界維持穩定。

---

## 9) 與現有研究之區別 (Positioning vs Prior Work)

1. **vs k-means / GMM / spectral**：
   它們以距離或線性代數結構定義 cluster；ELB 以密度誘導的能量拓撲（basin）定義 subclass，**邊界由 saddle/低密度分隔決定**，不是由距離等分決定。

2. **vs DBSCAN / HDBSCAN**：
   它們以「密度連通」做 heuristic 的 cluster；ELB 直接把 subclass 定義為梯度流的 stable manifold，並把穩定性條件（Morse/Morse–Smale）放進理論敘述。

3. **vs mean-shift / mode clustering**：
   mean-shift 可視為朝 (\nabla \log p) 上升，但多停在演算法層；ELB 強調 **(i) Morse index 的結構性分析、(ii) saddle 決定分界、(iii) 小擾動下 basin 拓撲穩定的定理化敘述**，使「subclass 是拓撲物件」成為可投稿的核心貢獻。

4. **vs TDA/Morse–Smale complex 的既有用法**：
   若既有工作把 Morse–Smale 用於可視化或低維分群，ELB 的差異在於：**把它定位為 subclass 的定義本體 + 與現代密度/score model 結合 + 提供可操作的高維 Hessian/index 估計與穩定性論證**。

---

## 10) Experiment 設計 (Experimental Design)

### 10.1 實驗 A：Toy 反例（展示「不是 cluster」）

* **資料**：兩高斯 + 中間低密度 ridge（或薄橋）
* **比較法**：k-means、GMM、spectral、DBSCAN/HDBSCAN、mean-shift
* **指標**：

  1. 與 ground-truth basin label 的一致性（ARI/NMI/accuracy）
  2. 邊界位置是否落在低密度區（boundary density score）
  3. 擾動穩定性：加噪/旋轉/小扭曲後 partition 變動率（VI / label flip rate）

### 10.2 實驗 B：高維影像 subclass（class-conditional）

* **流程**：先訓練一個 class-conditional score model（或用預訓練 encoder 後在 embedding 上做密度）
* **目標**：在同一 class 內找 subclass（風格、姿態、背景）
* **評估**：

  * 穩定性：不同 seed / 小擾動下 basin 指派一致性
  * 可解釋性：每個 basin 的代表樣本、沿 flow 的「語意變化」
  * 下游：以 subclass 作為輔助標籤做分類/校正，或做 OOD/shift 偵測（低密度邊界有利）

### 10.3 實驗 C：Saddle-aware boundary（驗證理論結構）

* **做法**：

  1. 找 index-1 saddle（score=0 + Hessian index=1）
  2. 從 saddle 附近沿不穩定方向追 separatrix
* **檢查**：

  * separatrix 是否對應實際 label flip 的邊界薄層
  * 邊界附近點的能量與密度是否符合「低密度分隔」假說
  * 擾動後 saddle 結構是否保持（index 不變、數量不大幅變化）

### 10.4 實驗 D：Multi-scale persistent subclass（加分項）

* **做法**：改變 smoothing/noise level，追蹤 basin merge/split
* **輸出**：subclass hierarchy / merge tree
* **評估**：是否能得到符合人類直覺的「粗到細」亞型階層，並提升穩定性（避免過度切分）。

---

如果你接下來要把它變成「可投 ML 理論/表示學習」的版本，我建議你下一步直接決定兩件事：

1. 你要在 *原始資料空間* 做 (p(x))，還是在 *learned representation (z=f(x))* 上做 (p(z))（兩者理論與實驗重點不同）；
2. 你要主打的 theorem 是「臨界點/index 穩定」還是「Morse–Smale 分解穩定」還是「score approximation → label flip 上界」。
   你不需要先問我，我也可以直接替你各做一版（偏理論 / 偏系統 / 偏應用）proposal。
