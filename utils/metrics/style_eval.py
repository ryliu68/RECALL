from transformers import pipeline
import torch
def style_classifier(device,path):
    device = torch.device("cuda:0")
    return pipeline('image-classification',model=path,device=device)

def style_eval(classifier,img):
    return classifier(img,top_k=129)