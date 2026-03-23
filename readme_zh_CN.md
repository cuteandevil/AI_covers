# AI 覆盖生成器

解决现有 SVC（歌声转换）和 RVC（基于检索的歌声转换）系统不足的工业级系统，实现全自动高质量伴奏音频生成。

## 特性

- **端到端自动化**：从原始音频输入到最终伴奏输出，无需人工干预。
- **多语言/多方言支持**：中文、英语等。
- **高质量语音合成**：输出接近人类歌声水平，机械伪迹最小化。
- **低延迟**：针对在线服务场景优化（推理延迟 <2秒）。
- **鲁棒性**：在不同录音条件和背景噪声下保持稳定性能。
- **自适应少样本/零样本说话人适应**：仅需 ≤3 分钟适应数据即可快速切换到新声音。
- **实时质量监控 & 自我纠正**：自动检测并修复伪迹。
- **边缘计算部署**：轻量级模型可通过 TensorRT/OpenVINO 部署在 CDN 节点上。
- **可选声源分离**：内置基于 Demucs 的声源/伴奏分离，使混合音频可作为输入（通过 `config.yaml` 启用）。启用时，最终输出将把转换后的人声与伴奏混合。
- **图形用户界面**：提供易用的 GUI，无需命令行知识即可生成伴奏。

## 架构

```
输入音频
    │
    ├─► [可选] 声源分离 (Demucs) → 人声
    │                                 （伴奏可选保存）
    │
    ├─► ASR (Whisper) → 歌词文本
    │
    ├─► 音频前端 → 基频, 能力, 说话人嵌入
    │
    └─► 多模态融合 (Conformer 编码器 + 注意力)
            │
            ▼
    自适应元学习器 (MAML 风格) → 个性化说话人参数
            │
            ▼
    神经声码器 (WaveNet/WaveGlow) → 波形生成
            │
            ▼
    质量监控 → 伪迹检测 & 自我纠正循环
            │
            ▼
    [可选] 混合伴奏（如果启用分离）
            │
            ▼
    输出伴奏音频
```

## 开始使用

1. **克隆仓库**（如果尚未完成）：
   ```bash
   git clone <repo-url> D:\AI_covers
   cd D:\AI_covers
   ```

2. **创建虚拟环境**（可选但推荐）：
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **下载预训练模型**（参见 `scripts/download_models.py` 或将它们放置在 `models/` 目录）：
   - Whisper ASR 模型 (`tiny.en` 或 `base`)
   - 说话人编码器（例如，ECAPA-TDNN）
   - F0 提取器（CREPE 或 Dio）
   - 神经声码器（WaveGlow 或 HiFi‑GAN）

5. **运行演示**：
   ```bash
   python main.py --input demo_input.wav --target_speaker example_speaker --output output_cover.wav
   ```

## 使用 GUI

项目还提供了一个图形用户界面（GUI）以实现更简便的交互。

1. 确保已安装项目依赖（参考 `requirements.txt` 和本 README）。
2. 如果使用虚拟环境，请激活它：
   ```bash
   cd D:\AI_covers
   venv\Scripts\activate
   ```
3. 启动 GUI：
   ```bash
   python gui.py
   ```
4. 在 GUI 窗口中：
   - 选择输入音频文件（您想要转换的歌曲）
   - 选择目标说话人音频（用作参考的参考声音）
   - 选择生成的伴奏音频保存位置
   - 配置可选设置（设备、伴奏保存、声源分离）
   - 点击 “Generate Cover” 开始生成
   - 在日志窗口中监控进度

## 配置

编辑 `config.yaml` 以调整：
- 模型和数据的路径
- ASR 语言
- 声码器类型
- 延迟/质量权衡
- 边缘部署标志

## 许可证

本项目采用 MIT 许可证。

## 致谢

- Whisper（OpenAI）
- ECAPA-TDNN 说话人嵌入
- CREPE 基频估计器
- WaveGlow / HiFi‑GAN 声码器
- MAML 用于少样本适应