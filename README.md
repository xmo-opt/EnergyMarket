# 日前电价预测

一个基于 **Python** 的示例项目，利用 **XGBoost** 回归模型在 15 分钟分辨率数据上预测日前电力市场未来 **96** 个结算点（24 小时）价格。

项目重点展示了如何在不发生时间信息泄漏的前提下进行特征工程、训练模型并保存预测结果。

---

## 主要特性

- **严格的时间边界处理**：所有滞后和滚动特征仅使用预测起始时间之前的数据。
- 自动生成价格、负荷、风电、光伏等 **滞后特征** 和 **滚动统计量**。
- 汇聚山东省八个城市（济南、潍坊、临沂、德州、滨州、泰安、烟台、青岛）的气象指标，形成平均气象特征。
- 预设 **XGBoost** 超参数，兼顾精度与训练速度。
- 训练后打印整体 **MAE / RMSE / R²**，并列出特征重要性。
- 将 96 点预测结果输出至 `day_ahead_price_forecast.csv`。

---

## 目录结构

```text
project/
├── dayaheadprice.py            # 主脚本
├── xytest.csv                  # 历史数据集（15 分钟分辨率）
└── README.md                   # 本文件
```

---

## 快速开始

### 环境安装

```bash
python3 -m venv venv
source venv/bin/activate        # Windows 请使用 venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt** 内容示例：

```
pandas
numpy
scikit-learn
xgboost
```

---

## 数据准备

脚本默认在工作目录查找名为 `xytest.csv` 的数据文件，最少包含下列表头：

| 列名                  | 含义                                   | 示例                  |
| ------------------- | ------------------------------------ | ------------------- |
| `times`             | 时间戳，ISO‑8601 或 `YYYY-MM-DD HH:MM:SS` | 2025-03-30 14:45:00 |
| `da_clearing_price` | 日前市场结算价                              | 420.31              |
| `rt_clearing_price` | 实时市场结算价                              | 415.22              |
| `load_actual`       | 系统负荷（MW）                             | 52 100              |
| `wind_actual`       | 风电出力（MW）                             | 6 230               |
| `solar_actual`      | 光伏出力（MW）                             | 4 870               |
| `济南_temp_2m`, …     | 每个城市的气象特征                            | 15.6                |

> **提示**：脚本会对数值列进行前向/后向填充，但完整数据将显著提升预测精度。

---

## 运行预测

```bash
python dayaheadprice.py
```

脚本流程：

1. 读取并清洗数据；
2. 在默认预测起点 `2025-04-05` 之前构造特征；
3. 训练 `XGBRegressor`；
4. 评估并将结果写入 `day_ahead_price_forecast.csv`。

控制台将显示整体误差指标及误差最大的 5 个小时段。

---

## 可配置项

- **预测起始时间**：在 `main()` 中修改 `prediction_start_time`。
- **预测长度**：调整 `evaluate_and_save()` 的 `forecast_horizon` 参数（默认 96）。
- **模型超参数**：在 `train_and_predict()` 中修改。

---

## 输出文件格式

`day_ahead_price_forecast.csv` 包含：

\| timestamp | actual\_price | forecast\_price | absolute\_error | percentage\_error |

---

## 可复现性

代码中固定 `random_state = 42`，以确保每次训练结果一致。去除该种子可获得随机化效果。

---

## 许可证

请在此处添加您选择的开源许可证（如 MIT）。

---

## 作者

**辛焱/ Yan Xin – XMO Decision（曦谋决策）**

欢迎提交 Issue 或 Pull Request 以改进项目。

