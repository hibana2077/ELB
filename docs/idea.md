## Research Proposal：Energy Landscape-Based Subclass（ELB）— 以 Morse 理論分層定義基因表現亞型

---

## 研究目標

1. **形式化定義「亞型（subclass）」為能量地形上的拓撲穩定 basin**：給定平滑密度 (p(x))，以能量 (E(x)=-\log p(x)) 的 **gradient flow** 之 **local minima 的 stable manifold** 作為亞型分區（而非距離式 cluster）。
2. **建立 ELB 的拓撲穩定性理論**：在小擾動（例如 batch effect、估計誤差、取樣噪音）下，證明 critical point 的 index、Morse–Smale 分層與 basin assignment 的穩定條件。Morse–Smale 系統的結構穩定性是核心支撐。 ([scholarpedia.org][1])
3. **把 ELB 落地於 microarray / gene expression subtype discovery**：在多個公開癌症基因表現資料集上，展示 ELB 能在「非球狀 / ridge / saddle 分隔」情境下，比傳統 clustering 更穩健地得到可重現、具生物意義的亞型分層。

---

## 預期貢獻

1. **Subclass 的新定義**：Subclass 被定義為能量地形的拓撲物件（basin / stratification），而不是距離相近的點集合。
2. **理論貢獻（可寫成 theorem 群）**：把「模式分群（mode clustering）」常見的 heuristic（如 mean-shift）提升為 **Morse–Smale/結構穩定** 的可證明框架；並把「擾動下 basin 不變」寫成明確條件。 
3. **方法論貢獻**：提出一個可操作的 ELB pipeline：高維基因表現 →（可解釋的）低維流形表示 → 平滑密度/能量估計 → Morse–Smale 分層與 persistence 簡化 → basin 指派與穩健性評估。
4. **生物學驗證貢獻**：提供「亞型穩健性（bootstrap 一致性）」+「臨床/生物效度（存活分離、已知 subtype 對齊、富集分析）」的系統化評估設計。

---

## 創新點

1. **不用 cluster，而用 dynamical system 的 attractor basin 定義 subclass**（basin-of-attraction 作為 label 生成機制）。
2. **使用 Morse–Smale complex 作為分層骨架**：不只看 minima（mode），還顯式利用 saddle 結構（Hessian signature / separatrix）來刻畫 subtype 邊界與可合併尺度（persistence）。 ([pub.ista.ac.at][2])
3. **以「拓撲穩定」取代「距離假設」**：對 microarray 常見的高噪音、小樣本、非線性分隔，主張用拓撲不變性作為 subclass 的本質。

---

## 理論洞見

1. **亞型邊界本質上是 separatrix（由 saddle 的 stable/unstable manifold 決定）**：傳統 clustering 常把「低密度 ridge / saddle 分隔」誤當成可忽略噪音；ELB 反而把它視為 subtype 的拓撲邊界。
2. **多尺度亞型是自然產物**：藉由 Morse–Smale complex 的 persistence-based simplification，可得到由粗到細的亞型層級（類似 cluster tree，但有明確的拓撲意義與取消（cancellation）規則）。 ([graphics.stanford.edu][3])
3. **高維未必毀掉 mode/basin 思路**：已有理論指出 mode-based clustering 的風險可以在「高密度 core」區域很小，且可在低噪音條件下控制整體風險；ELB 會把這條線推進到「Morse 分層 + 穩定性」層次。 

---

## 理論貢獻（可寫進論文的核心命題）

### 命題 A：非退化 critical point 與 index 的擾動穩定

若 (E\in C^2) 為 Morse function（所有 critical point 非退化），且 (|\widehat{E}-E|_{C^2}) 足夠小，則每個 critical point (c) 對應到一個唯一 (\hat c)（位置小偏移），且 Hessian signature（index）保持不變。

*證明路線（摘要）*：對 (\nabla E(c)=0)、(\det \nabla^2E(c)\neq 0) 用 **implicit function theorem**；index 不變由 Hessian 連續性推出。

### 命題 B：Morse–Smale 分層的結構穩定與 basin 不變性

若 (\dot x = -\nabla E(x)) 為 Morse–Smale gradient-like flow，則系統在小擾動下具有結構穩定性；因此 stable manifold 分割（basin 分區）在拓撲意義下保持一致（存在同胚把一個分割送到另一個分割）。 ([scholarpedia.org][1])

### 命題 C：估計誤差 (\Rightarrow) basin 指派風險界

令 (\widehat{E}) 由估計的 (\widehat p) 得到（或由 score model/flow 得到可微 (\log \widehat p)），若 (|\widehat{E}-E|_{C^2}) 有上界，則 basin assignment 的錯誤主要集中在「靠近分界面（saddle separatrix）」的薄層；可用「分界面 (\delta)-neighborhood 的機率質量」上界錯分率，並可對應到已知的 mode clustering 風險分析與 flow 穩定性結果。 

---

## 方法論

### 1) 資料前處理（microarray 特化）

* 以 GEO/Curated 來源取得 expression matrix（建議先用已標準化的 series matrix / harmonized matrices，降低 RMA/CEL 門檻）。
* 常見處理：(\log_2) transform、quantile normalization（若來源未做）、probe→gene 彙整、低變異基因過濾、批次校正（ComBat 等，作為「擾動測試」的一部分）。

### 2) 表徵空間（避免高維密度估計崩壞）

* 先做可解釋降維：PCA（保留 20–100 維）、或帶可逆/平滑性約束的 autoencoder latent。
* ELB 的正式定義可在 latent manifold (z=g(x)) 上做：學 (p_Z(z))，設 (E(z)=-\log p_Z(z))。

### 3) 平滑密度 / 能量估計（兩條可投稿的路線）

* **Normalizing Flow**（可得 exact/tractable (\log p)）：例如 RealNVP 類模型，便於直接算 (\nabla E)、Hessian。 ([arXiv][4])
* **Score-based model（SDE diffusion）**：直接學 score (\nabla \log p)，再定義 flow (\dot z = -\nabla E(z)=\nabla \log p(z))，可用現成框架。 ([arXiv][5])

### 4) Morse–Smale 分層與 basin 指派（連續 + 離散兩版）

* **連續版（autograd）**：多起點做 gradient flow 收斂到 minima；在 minima 估 Hessian eigenvalues 確認 index=0。
* **離散版（更實務，適合高維小樣本）**：在 kNN 圖上定義離散能量場，做 steepest descent 到局部 minima，並以離散 Morse–Smale complex 概念建立分區，再做 persistence 簡化移除低顯著 minima（去噪、提升穩健）。 

---

## 數學理論推演與證明（寫作建議結構）

1. **定義（ELB subclass）**
   給 (p(x)>0) 平滑，(E(x)=-\log p(x))。gradient flow
   [
   \dot x = -\nabla E(x)
   ]
   對每個 local minimum (m_k)，定義 basin（stable manifold）
   [
   \mathcal B_k = {x: \lim_{t\to\infty}\phi_t(x)=m_k}.
   ]
   Subclass (k := \mathcal B_k)。

2. **Morse–Smale 條件與分層（stratification）**
   要求 (E) 為 Morse 且 stable/unstable manifolds 橫截（transverse），得到 Morse–Smale complex。 ([scholarpedia.org][1])

3. **穩定性證明主線**

   * critical point 持續性：implicit function theorem（命題 A）。
   * flow 的穩定性：已知結果可把「函數 (C^2) 接近」轉換為「gradient flow 線與終點接近」，並可推導 basin 邊界擾動界。 
   * 結構穩定性：Morse–Smale 系統對小擾動具結構穩定。 ([scholarpedia.org][1])

4. **統計層（finite sample）對接**

   * 用 (|\widehat{E}-E|_{C^2}) 控制 basin 變化。
   * 參考 mode clustering 風險分析：錯分集中在分界薄層；高密度 core 區錯分可極小。 

---

## 預計使用資料集（Gene Expression / Microarray；可用 Python 自動下載）

### A. 直接用 NCBI GEO（GEOparse 一行下載）

* **GSE13159（MILE leukemia, 2096 samples）**：大型白血病診斷/亞型資料集，適合測 ELB 在多類別與複雜分隔下的 basin 結構。 ([國家生物技術資訊中心][6])
* **GSE2034（breast cancer relapse-free survival）**：含遠端轉移/復發資訊，適合評估 ELB subclass 的預後分離。 ([國家生物技術資訊中心][7])
* **GSE1456（breast tumors, n=159）**：常用預後/分類資料集，可做跨資料集穩健性測試。 ([國家生物技術資訊中心][8])
* **GSE45827（breast cancer subtypes）**：對 subtype discovery 很直接（可用既有 subtype label 做外部驗證）。 ([國家生物技術資訊中心][9])

**Python 下載示例（GEOparse）**：GEOparse 提供下載與解析 GEO Series / Samples / Platforms 的標準介面。 ([PyPI][10])

```python
import GEOparse
gse = GEOparse.get_GEO("GSE2034", destdir="./geo")
# gse.gsms: samples；gse.gpls: platform annotation
```

### B. 使用 refine.bio（已 harmonize 的 expression matrix + metadata；有 Python client）

refine.bio 可把公開轉錄體資料統一處理並輸出可直接 ML 的矩陣與樣本中繼資料，並提供 Python client。 ([GitHub][11])

```python
import pyrefinebio
# 依 refine.bio API 建 dataset，並下載 zip（包含 expression matrix 與 metadata）
```

### C. CuMiDa（癌症 microarray benchmark 合集）

CuMiDa 提供「篩選+正規化+品質控管」後的 78 個人類癌症 microarray 資料集，定位就是 ML benchmarking；可作為 ELB 的廣泛基準測試池。 ([PubMed][12])

---

## 與現有研究之區別

1. **相對於 mean-shift / mode clustering**

   * 傳統模式分群把 cluster 視為 mode 的 basin-of-attraction，但多停留在演算法或風險分析，且常用 KDE + heuristic mean-shift。 
   * ELB 的差異：把 subclass 定義提升到 **Morse–Smale stratification**，並把 saddle 結構（邊界）與 persistence（多尺度）納入定義與穩定性證明骨架。 ([pub.ista.ac.at][2])

2. **相對於 topological ML（如用 Mapper/PH 做聚類）**

   * ELB 的拓撲單元不是「連通性/覆蓋」而是 **gradient-flow 誘導的分層**；subclass 直接對應到動力系統吸引域，理論上可用結構穩定性處理擾動。 ([scholarpedia.org][1])

3. **相對於既有 Morse–Smale 在統計的應用（如 Morse–Smale regression）**

   * 既有工作用 Morse–Smale complex 做回歸分段與 persistence 簡化；ELB 則把它用於「亞型定義與穩健分群」，並將密度/能量估計誤差與 basin 穩定性系統連結（命題 A–C）。 ([PMC][13])

---

## Experiment 設計

### 1) Toy experiment（驗證「不是距離問題，是地形問題」）

* 生成：兩群高斯 + 中間低密度 ridge（或多鞍點結構）。
* 比較：k-means/GMM/spectral/DBSCAN vs mean-shift vs ELB（含 persistence 簡化）。
* 指標：對真實 basin label 的 ARI/NMI；對擾動（加噪、改帶寬、減樣本）的一致性曲線。

### 2) Real data：Subtype discovery（核心主實驗）

對每個 GEO dataset（GSE13159 / GSE2034 / GSE1456 / GSE45827）：

**(a) 外部效度（若有 label）**

* 用已知 subtype/診斷標籤：ARI/NMI、macro-F1（把 subclass 當 pseudo-label 做 downstream classifier）。
* 對 GSE45827 這類 subtype 資料集，測 ELB 是否能在不指定 (K) 的情況下恢復主要亞型。

**(b) 臨床效度（若有 survival/relapse）**

* 在 GSE2034（relapse-free）上：Kaplan–Meier 分離、log-rank test、CoxPH hazard ratio（用 subclass 當分層變數）。

**(c) 生物可解釋性**

* 每個 basin 做 differential expression（limma/簡化版 t-test），做 GO/Pathway 富集；檢查 basin 是否對應已知生物機制（例如免疫浸潤、ER/HER2 軸、細胞週期）。

### 3) Robustness / Stability（你的理論要落地成圖）

* **Bootstrap stability**：重抽樣本、重抽基因子集（feature bagging），看 subclass assignment 的一致性（平均 ARI）。
* **Batch-like perturbation**：加入平台/批次偏移（或用 ComBat 前後比較），檢查「critical point 數量與 persistence hierarchy」是否穩定。
* **Ablation**：

  1. 密度估計器（KDE vs flow vs score-based） ([arXiv][4])
  2. latent 維度
  3. persistence 閾值（取消低顯著 critical pairs） 

---

如果你要把它寫成可投稿的「理論+實證」論文，建議主線就是：**定義（ELB）→ Morse–Smale 結構 → 擾動穩定 theorem（A/B/C）→ 離散近似演算法（含 persistence）→ GEO/微陣列 subtype 的穩健性與臨床效度**。

[1]: https://www.scholarpedia.org/article/Morse-Smale_systems?utm_source=chatgpt.com "Morse-Smale systems"
[2]: https://pub.ista.ac.at/~edels/Papers/2003-08-MorseSmaleComplexes3D.pdf?utm_source=chatgpt.com "Morse-Smale Complexes for Piecewise Linear 3-Manifolds"
[3]: https://graphics.stanford.edu/courses/cs468-01-fall/Papers/edelsbrunner_harer_zomorodian.pdf?utm_source=chatgpt.com "Hierarchical Morse Complexes for Piecewise Linear 2- ..."
[4]: https://arxiv.org/abs/1605.08803?utm_source=chatgpt.com "[1605.08803] Density estimation using Real NVP"
[5]: https://arxiv.org/abs/2011.13456?utm_source=chatgpt.com "Score-Based Generative Modeling through Stochastic Differential Equations"
[6]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159&utm_source=chatgpt.com "GSE13159 - GEO Accession viewer - NIH"
[7]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=gse2034&utm_source=chatgpt.com "Series GSE2034 - GEO Accession viewer - NIH"
[8]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=gse1456&utm_source=chatgpt.com "GSE1456 - GEO Accession viewer - NIH"
[9]: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827&utm_source=chatgpt.com "GSE45827 - GEO Accession viewer - NIH"
[10]: https://pypi.org/project/GEOparse/ "GEOparse · PyPI"
[11]: https://github.com/AlexsLemonade/refinebio-py?utm_source=chatgpt.com "AlexsLemonade/refinebio-py: A python client for the refine. ..."
[12]: https://pubmed.ncbi.nlm.nih.gov/30789283/?utm_source=chatgpt.com "CuMiDa: An Extensively Curated Microarray Database for ..."
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3653333/?utm_source=chatgpt.com "Morse-Smale Regression - PMC - NIH"
