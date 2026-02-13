

# **方向二：Energy Landscape-Based Subclass via Morse-Theoretic Stratification**
ELB

（用能量地形與 Morse 理論定義亞型）

---

## 🎯 核心想法

不把 subclass 定義為 cluster。

定義為：

> 在樣本密度函數所形成的能量地形中的「拓撲穩定 basin」。

也就是：

* 學習一個 smooth density estimator ( p(x) )
* 定義能量 ( E(x) = -\log p(x) )
* 分析其 Morse critical points
* 每個 basin of attraction = 一個 subclass

這是從 dynamical systems 定義 subclass。

---

## 📐 理論基礎

* Morse theory
* Gradient flow
* Topological stratification
* Critical point theory

傳統 clustering：

* 依賴距離
* 依賴 mixture

這裡：

> 依賴能量地形的拓撲不變性

Subclass 是拓撲物件，不是距離物件。

---

## 🧮 數學定義

給定 smooth density ( p(x) )

[
E(x) = -\log p(x)
]

考慮 gradient flow：

[
\frac{dx}{dt} = -\nabla E(x)
]

不同初始點收斂到不同 local minima

定義：

[
\text{Subclass}*k = {x : \lim*{t→∞} \phi_t(x) = m_k }
]

其中 ( m_k ) 是 local minimum。

---

## 🔬 為什麼新？

Mean-shift、density clustering 是 heuristic。

這裡：

* 嚴格用 Morse theory
* 利用 Hessian signature 分析 saddle structure
* 用拓撲穩定性證明 subclass 的穩定條件

你可以給出 theorem：

> 若 density perturbation 小於 ε，critical point index 不變 → subclass 穩定

這是理論性 contribution。

---

## 🧪 Toy Experiment

生成資料：

* 兩群高斯，但中間加一條低密度 ridge
* 傳統 k-means 會錯分
* 能量 flow 會正確分 basin

實作：

* 用 PyTorch 建立 score-based model
* 用 autograd 計算 Hessian
* 用 gradient flow 模擬收斂 basin

---

## 🧠 Insight

Subclass 不是 cluster。

Subclass 是：

> density landscape 的穩定 attractor。

這讓 subclass 定義具有：

* 拓撲穩定性
* 可證明 robustness
* 可與 dynamical system 理論連結

---

# 兩方向比較

|             | OT-field subclass      | Morse energy subclass  |
| ----------- | ---------------------- | ---------------------- |
| subclass 定義 | transport map topology | energy basin           |
| 理論基底        | Optimal Transport      | Morse theory           |
| 本質          | 幾何變形                   | 動力系統穩定態                |
| 與現有方法差異     | 不在樣本空間分群               | 不在距離空間分群               |
| PyTorch 可行性 | Sinkhorn + NN map      | Score model + autograd |

---

# 哪個更適合做 Analysis-type ML paper？

若你想做「偏理論 + 嚴謹數學」：

👉 Morse-theoretic 方向更強。

若你想做「幾何深度學習 + representation」：

👉 OT-field 方向更有潛力。

---

如果你願意，我可以幫你：

* 推導 formal theorem statement
* 設計完整 toy experimental protocol
* 給出可投稿的論文架構草稿
* 或幫你分析哪個方向更可能中 ICML / NeurIPS 理論軌

告訴我你想走哪條路。
