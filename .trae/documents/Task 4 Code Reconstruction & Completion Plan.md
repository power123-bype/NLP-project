# 完善 Task 4 融合模型训练与推理流程计划

根据您的要求，我将重构并完善代码以完成 Task 4。虽然目前的 `train.py` 已经可以运行，但为了确保代码的健壮性和可复用性（特别是推理阶段），我建议执行以下步骤：

## 1. 重构 Task 3 代码 (`task3.py`)
目前的 `task3.py` 是一个用于批量生成的 Colab 脚本，包含非必要的 Colab 依赖，且难以被推理脚本直接调用。
- **修改目标**：
    - 移除 `google.colab` 等无关依赖。
    - 封装核心功能为 `get_text_embedding(text)` 函数，使其可以像 `task2.py` 那样被外部调用。
    - 确保使用 `distilbert-base-uncased` 模型，与训练数据保持一致。

## 2. 改进训练脚本 (`train.py`)
目前的训练脚本功能较基础。
- **修改目标**：
    - **增加验证集划分**：将数据按 8:2 划分为训练集和验证集，用于监控模型是否过拟合。
    - **保存最佳模型**：仅在验证集 Loss 降低时保存模型（`best_model.pth`），而不仅仅是保存最后一个 Epoch。
    - **优化代码结构**：确保正确引用重构后的 `task2` 和 `task3`（尽管训练主要用离线数据，但保持一致性很好）。

## 3. 完善推理脚本 (`inference.py`)
构建一个端到端的推理演示，证明 Task 4 完成。
- **功能**：
    - 输入：一张图片路径 + 一句文本指令。
    - 过程：
        1. 调用 `task2.get_image_embedding` 获取图片特征。
        2. 调用 `task3.get_text_embedding` 获取文本特征。
        3. 调用 `fusion_model` 进行预测。
    - 输出：预测的 `brightness`, `contrast`, `temp` 调整参数。

## 4. 执行验证
- 运行改进后的 `train.py` 进行完整训练。
- 运行 `inference.py` 对测试图片进行推理，展示结果。

---

**确认事项**：
我已理解您提到的数据路径映射（Task 2 -> NLP, Task 3 -> CV）是口误，实际上 **Task 2 (ResNet) 对应 CV Embedding**，**Task 3 (DistilBERT) 对应 NLP Embedding**。我将按照代码实际逻辑（3870张图片 vs 19355条文本）进行处理。

请确认是否开始执行此计划？