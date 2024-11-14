import torch
import torch.nn as nn
import torch_neuronx
import numpy as np
import base64
import traceback
import logging
import os
import time
import copy
from io import BytesIO

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CLIPImageProcessor
from transformers_neuronx import MistralForSampling, GQA, NeuronConfig, QuantizationConfig

from utils import load_image,load_images,unpad_image,tokenizer_image_token,prepare_inputs_labels_for_multimodal

device=torch.device('cpu')

DEFAULT_IMAGE_TOKEN='<image>'
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX=-200
HEIGHT=WIDTH=336
VISION_TOWER_NAME='openai/clip-vit-large-patch14-336'
MODEL_BASE_PATH='./neuron/'
model_path=MODEL_BASE_PATH+"mistral-7b"

model_vit = torch.jit.load(MODEL_BASE_PATH+"neuron-model-clip-vit.pt")
model_vit.eval()

model_projector = torch.jit.load(MODEL_BASE_PATH+"neuron-model-mm-projector.pt")
model_projector.eval()

image_newline = torch.load(MODEL_BASE_PATH+'image_newline.bin',map_location=device)

pretrained_weight = torch.load(MODEL_BASE_PATH+'embed_tokens.bin',map_location=device)
embed_tokens=nn.Embedding(32000, 4096)
embed_tokens.load_state_dict(pretrained_weight, strict=False)


if not torch.cuda.is_available():
    model_mistral = MistralForSampling.from_pretrained(
        model_path,
        map_location=device,
        amp='bf16',
        batch_size=1,
        tp_degree=2,
        n_positions=4096,
    )
neuron_path=MODEL_BASE_PATH+'bs1_tp2/'
model_mistral.load(neuron_path)
model_mistral.to_neuron()

tokenizer = AutoTokenizer.from_pretrained(model_path)



def generate(inputs,image_files):
    #image_files="https://llava-vl.github.io/static/images/view.jpg"
    images = [load_image(image_files)]

    t0=time.time()
    #byte_data = base64.b64decode(byte_image)
    #images = [Image.open(BytesIO(byte_data)).convert('RGB')]

    #inputs = "What are the things I should be cautious about when I visit here?"
    qs = DEFAULT_IMAGE_TOKEN + "\n" + inputs
    temperature=0.2
    top_p=0.1
    top_k=100
    max_new_tokens=256
    n_positions = 4096

    prompt="[INST] {} [/INST]".format(qs)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    image_sizes = [x.size for x in images] #[(1000, 667)]
    image_sizes = [(1000, 667)]

    image_new=images[0].resize((WIDTH, HEIGHT))

    image_processor = CLIPImageProcessor.from_pretrained(VISION_TOWER_NAME)

    image_patch = image_processor.preprocess(image_new, return_tensors='pt')['pixel_values'][0]

    image_patchs=image_patch.repeat(5,1,1,1)

    with torch.inference_mode():
        image_features = model_vit(image_patchs)
        image_features=image_features['hidden_states'][:, 1:]

        image_features = model_projector(image_features)

        new_input_embeds=prepare_inputs_labels_for_multimodal(image_features,input_ids, None, None, None, None, None, image_sizes,embed_tokens,image_newline)

        input_embeds_tensor=new_input_embeds
        t1=time.time()
        logging.info(f"handle: :Preparation TIME TAKEN: { (t1 - t0) * 1000} ms:::")
        output_ids = model_mistral.sample(input_embeds_tensor, sequence_length=n_positions, start_ids=None, top_k=top_k, top_p=top_p, temperature=temperature, )
        outputs = [tokenizer.decode(tok) for tok in output_ids]
        result = outputs[0].replace('</s>','')
        logging.info(f"handle: :Generation TIME TAKEN: { (time.time() - t1) * 1000} ms:::")
        return result

prompt ="What are the things I should be cautious about when I visit here?"
image_path="https://llava-vl.github.io/static/images/view.jpg"

result = generate(prompt,image_path)
print(result)
