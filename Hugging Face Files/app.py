### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from predictor import predictionMaker

exampleList = [["examples/" + example] for example in os.listdir("examples")]

title = "Food Vision👀 on Food101 Using ViT"
description = "Trained a ViT to classify images of food based on [Food101](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)."
article = "Created by [Eternal Bliassard](https://github.com/EternalBlissard)."

# Create the Gradio demo
demo = gr.Interface(fn=predictionMaker, 
                    inputs=[gr.Image(type="pil")], 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=exampleList, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch() 




