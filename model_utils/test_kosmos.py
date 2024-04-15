import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from . import get_image



class TestKosmos:
    def __init__(self, device=None):
        pass
        
    def move_to_device(self, device=None):
        pass
    @torch.no_grad()
    def generate(self, image, question):
        pass
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list):
        pass
