### Imports for Modules ### 
import gradio as gr
import os
import torch
from typing import Tuple, Dict
from timeit import default_timer as timer

### Functional Imports
from model import getViT

classNames = ['Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare', 'Beet Salad', 'Beignets', 'Bibimbap', 'Bread Pudding', 'Breakfast Burrito', 'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 'Carrot Cake', 'Ceviche', 'Cheese Plate', 'Cheesecake', 'Chicken Curry', 'Chicken Quesadilla', 'Chicken Wings', 'Chocolate Cake', 'Chocolate Mousse', 'Churros', 'Clam Chowder', 'Club Sandwich', 'Crab Cakes', 'Creme Brulee', 'Croque Madame', 'Cup Cakes', 'Deviled Eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs Benedict', 'Escargots', 'Falafel', 'Filet Mignon', 'Fish And Chips', 'Foie Gras', 'French Fries', 'French Onion Soup', 'French Toast', 'Fried Calamari', 'Fried Rice', 'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad', 'Grilled Cheese Sandwich', 'Grilled Salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot And Sour Soup', 'Hot Dog', 'Huevos Rancheros', 'Hummus', 'Ice Cream', 'Lasagna', 'Lobster Bisque', 'Lobster Roll Sandwich', 'Macaroni And Cheese', 'Macarons', 'Miso Soup', 'Mussels', 'Nachos', 'Omelette', 'Onion Rings', 'Oysters', 'Pad Thai', 'Paella', 'Pancakes', 'Panna Cotta', 'Peking Duck', 'Pho', 'Pizza', 'Pork Chop', 'Poutine', 'Prime Rib', 'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed Salad', 'Shrimp And Grits', 'Spaghetti Bolognese', 'Spaghetti Carbonara', 'Spring Rolls', 'Steak', 'Strawberry Shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna Tartare', 'Waffles']
ViTModel, VitTransforms = getViT(42,classNames,torch.device("cpu"))
ViTModel.load_state_dict(torch.load(f="ViTModel.pt",map_location=torch.device("cpu")))

def predictionMaker(img):
  startTime = timer()
  img = VitTransforms(img).unsqueeze(0)
  ViTModel.eval()
  with torch.inference_mode():
    predProbs = torch.softmax(ViTModel(img),dim=1)
  predDict = {classNames[i]: float(predProbs[0][i]) for i in range(len(classNames))}
  endTime = timer()
  predTime = round(endTime-startTime,4)
  return predDict,predTime
