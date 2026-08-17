"""
文件作用：启动作品集 Gradio 页面，展示 Top-3 预测与 Grad-CAM。
File purpose: launch the interactive portfolio demo.
"""

from __future__ import annotations

import argparse

import gradio as gr
import torch
from PIL import Image

from src.vision.inference import Predictor
from src.vision.visualization import GradCAM, overlay_heatmap


def build_app(checkpoint_path: str) -> gr.Interface:
    """加载一次 checkpoint 并创建可复用的 Gradio Interface。"""

    predictor = Predictor(checkpoint_path)

    def classify(image: Image.Image):
        """返回 Gradio Label 字典与模型关注区域。"""

        if image is None:
            return {}, None
        prediction = predictor.predict(image, top_k=3)
        labels = dict(zip(prediction.labels, prediction.scores))
        tensor = predictor.transform(image.convert("RGB")).unsqueeze(0).to(predictor.device)
        cam = GradCAM(predictor.model)
        try:
            heatmap = cam.generate(tensor, prediction.predicted_index)
        finally:
            cam.close()
        return labels, overlay_heatmap(image, heatmap)

    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Bird image"),
        outputs=[
            gr.Label(num_top_classes=3, label="Top-3 predictions"),
            gr.Image(type="pil", label="Grad-CAM explanation"),
        ],
        title="CUB-200 Fine-Grained Bird Classification",
        description="ConvNeXt/ResNet/Custom CNN checkpoint inference with Top-3 confidence and Grad-CAM.",
    )


def main() -> None:
    """解析 checkpoint 参数并启动本地 Web UI。"""

    parser = argparse.ArgumentParser(description="Launch the CUB portfolio demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true", help="Ask Gradio for a temporary share link")
    args = parser.parse_args()
    build_app(args.checkpoint).launch(share=args.share)


if __name__ == "__main__":
    main()

