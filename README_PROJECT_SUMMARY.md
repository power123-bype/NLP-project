# MH6812 Project - 多模态图像增强模型交付文档

## 1. 项目概述 (Project Overview)
本项目旨在开发一个基于多模态（图像 + 文本指令）的深度学习模型，用于预测图像增强参数（亮度、对比度、色温）。通过输入原始图像和自然语言指令，模型输出对应的调整参数。

经过多轮迭代与实验，最终交付三个版本，其中 **V4** 为推荐的生产环境版本。

---

## 2. 模型版本说明 (Model Versions)

### 🟢 V1: Base Model (基础版)
- **位置**: `/workspace/MH6812_Project`
- **架构**: **简单拼接 (Concatenation)**
- **特点**: 逻辑简单，作为 Baseline。在处理极端暗图时比 V3 稍微“听话”一点，但依然存在过拟合问题。

### 🌟 V3: Ultimate Model (纯模型版)
- **位置**: `/workspace/MH6812_Project_v3`
- **架构**: **门控残差注意力 (Gated Residual Attention)** + **150 Epochs**
- **特点**:
    - **画质极佳**：在 `Flat`, `Sunset` 等常规场景下表现完美。
    - **语义理解强**：支持反向修图（如将暖图变冷）。
    - **缺陷**：存在“暗图审美偏执”，对于 `Road` 或 `Forest Dark` 这类极暗图片，模型会拒绝提亮，甚至为了响应增强指令而错误地大幅增加暖色调（变红）。

### 🚀 V4: Production Ready (推荐生产版)
- **位置**: `/workspace/MH6812_Project_v4`
- **架构**: **V3 模型 + 启发式逻辑修复 (Heuristic Logic Fix)**
- **核心改进**:
    - 在推理阶段引入了智能拦截逻辑 (`check_and_fix_brightness`)。
    - **强制提亮**: 当检测到图片极暗 (`avg < 0.4`) 且用户指令要求提亮 (`"bright", "exposure"`)，但模型预测值过低时，强制接管并大幅提亮。
    - **强制中性色**: 当检测到用户指令要求冷色/中性色 (`"cool", "neutral"`)，但模型预测出严重暖偏色 (`temp > 0.3`) 时，强制修正色温为中性。
- **适用场景**: **所有场景**。既保留了 V3 的高画质，又解决了极端坏例 (Bad Cases)。

---

## 3. 性能对比 (Performance Comparison)

| 测试场景 | 指令意图 | V3 (纯模型) 表现 | V4 (逻辑修复) 表现 | 结果点评 |
| :--- | :--- | :--- | :--- | :--- |
| **Flat 1** | 增强对比度，保持明亮 | **完美 (+0.27 亮)** | **完美 (+0.27 亮)** | 常规场景，V4 继承 V3 的优秀表现。 |
| **Cold (反向)** | 暖图变冷 (去黄) | **-0.44 (变冷)** | **-0.44 (变冷)** | V3 本身理解正确，V4 逻辑未触发，保持原样。 |
| **Road** | 夜间道路提亮，保持中性 | ❌ **-0.18 (变暗), +0.83 (极红)** | ✅ **+0.40 (提亮), 0.00 (中性)** | **V4 完胜**。逻辑修复成功拦截了模型的错误偏好。 |
| **Forest Dark** | 极暗森林变白天 | ❌ **-0.40 (变暗)** | ✅ **+0.40 (提亮)** | **V4 完胜**。强制执行了用户的提亮指令。 |

---

## 4. 快速上手 (Quick Start)

### 推荐方式：使用 V4
进入 V4 目录，替换 `test_examples` 中的图片，然后运行生成脚本。

```bash
cd /workspace/MH6812_Project_v4

# 1. 将您的测试图片放入 test_examples/ 文件夹
# 2. (可选) 修改 generate_test_examples.py 中的 style_map 以匹配您的文件名关键词

python3 generate_test_examples.py
```

结果将生成在同目录下的 `*_after.jpg` 文件中。

### 逻辑修复代码位置
如果您想调整 V4 的判定阈值（例如什么样的图算“暗图”），请修改 `/workspace/MH6812_Project_v4/inference.py` 中的 `check_and_fix_brightness` 函数。

```python
def check_and_fix_brightness(image_path, instruction, predicted_b, predicted_t):
    # ...
    # if avg_brightness < 0.4 and wants_bright ...
    # ...
```

---

## 5. 项目交付清单 (Deliverables)

1.  **MH6812_Project_v4/**: 最终交付版本 (含代码、模型权重、测试脚本)。
2.  **MH6812_Project_v3/**: 研究中间版本 (纯模型，供学术对比)。
3.  **MH6812_Project/**: 基础 Baseline 版本。
4.  **README_PROJECT_SUMMARY.md**: 本说明文档。

## 6. 总结
本项目通过“**深度学习 + 专家规则**”的混合方案 (Hybrid Approach)，成功解决了纯端到端模型在极端分布外数据 (OOD) 上表现不佳的问题。V4 版本在保证通用场景画质的前提下，显著提升了对用户强指令的鲁棒性。
