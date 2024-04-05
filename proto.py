import argparse
from PIL import Image
import torch
import json
import os
from tqdm import tqdm
import pandas as pd

models = ['blip2', 'instructblip', 'llava', 'git', 'kosmos2', 'vilt']

anno = pd.read_csv('output/questions.csv')
    
def getBLIP2(device):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl").to(device)
    template = "Question: {} Answer:"
    
    def forward(model, processor, image, text):
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        inputs['max_length'] = 51
        generated_ids = model.generate(**inputs)
        final = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return final
    return model, processor, template, forward
    
def getInstructBLIP(device):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl").to(device)
    template = "{}"
    
    def forward(model, processor, image, text):
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        inputs['max_length'] = 51
        outputs = model.generate(**inputs)
        final = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return final
    return model, processor, template, forward
    
def getLLAVA(device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf").to(device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")
    template = "<image>\nUSER: {}\nASSISTANT:"
    
    def forward(model, processor, image, text):
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        inputs['max_length'] = 51
        generate_ids = model.generate(**inputs)
        final = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return final
    return model, processor, template, forward
    
def getGIT(device):
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa").to(device)
    template = "{}"
    
    def forward(model, processor, image, text):
        pixel_values = processor(images=image, return_tensors="pt").to(device).pixel_values
        input_ids = processor(text=text, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=51)
        final = processor.batch_decode(generated_ids, skip_special_tokens=True)[0][len(txt)+1:]
        return final
    return model, processor, template, forward
    
def getKOSMOS2(device):
    from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    template = "Question: {} Answer:"
    
    def forward(model, processor, image, text):
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(pixel_values=inputs["pixel_values"],
                                       input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"],
                                       image_embeds=None,
                                       image_embeds_position_mask=inputs["image_embeds_position_mask"],
                                       use_cache=True,
                                       max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
        final, _ = processor.post_process_generation(generated_text)
        return final
    return model, processor, template, forward
    
def getVILT(device):
    from transformers import ViltProcessor, ViltForQuestionAnswering
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    template = "{}"
    
    def forward(model, processor, image, text):
        encoding = processor(image, text, return_tensors="pt").to(device)
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return model.config.id2label[idx]
    return model, processor, template, forward
    
_mfuncdict = {'blip2': getBLIP2,
              'instructblip': getInstructBLIP,
              'llava': getLLAVA,
              'git': getGIT,
              'kosmos2': getKOSMOS2,
              'vilt': getVILT}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate VQ answers for a given model.')
    parser.add_argument('-m', '--model', type=str, default='blip2', choices = models,
                        help='name of model to generate answers')
    parser.add_argument('-ipth', '--imgpth', type=str, default='laion',
                        help='path of image dataset to use')
    parser.add_argument('-gpu', '--usegpu', action='store_true',
                        help='path of image dataset to use')                    
    args = parser.parse_args()
    
    device = 'cpu'
    if args.usegpu:
        print('using GPU...')
        device = 'cuda'
    
    mfunc = _mfuncdict.get(args.model)
    model, processor, template, forward = mfunc(device)
    imgdir = args.imgpth
    
    ansdict = {}
        
    for quesid, detail in anno.iterrows():
        imgpth = detail[0]
        question = detail[1]
        print(quesid,',',question)
        txt = template.format(question)
        
        img = Image.open(os.path.join(imgdir, imgpth)).convert('RGB')
        
        final = forward(model, processor, img, txt)
        print(final)
        print() 
        ansdict[quesid] = {'img': imgpth, 'ans': final}
        
    with open('output/'+args.model+'_ans.json', 'w') as f:
        json.dump(ansdict, f)
