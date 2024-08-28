from djl_python.inputs import Input
from djl_python.outputs import Output
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
from huggingface_hub import snapshot_download

from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CLIPImageProcessor
from transformers_neuronx import MistralForSampling, GQA, NeuronConfig, QuantizationConfig

from utils import load_image,load_images,unpad_image,tokenizer_image_token,prepare_inputs_labels_for_multimodal

os.environ['NEURON_CC_FLAGS'] = f"--enable-internal-io-dge"
device=torch.device('cpu')

DEFAULT_IMAGE_TOKEN='<image>'
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX=-200
HEIGHT=WIDTH=336
VISION_TOWER_NAME='openai/clip-vit-large-patch14-336'
MODEL_BASE_PATH='/opt/ml/model/'

model_vit=None
model_projector=None
model_mistral=None
image_newline=None
embed_tokens=None
tokenizer=None

def load_model(properties):
    global model_vit
    global model_projector
    global model_mistral
    global image_newline
    global embed_tokens
    global tokenizer
    tensor_parallel = properties["tensor_parallel_degree"]
    MODEL_BASE_PATH = properties['model_dir']
    if "model_id" in properties: #s3 path
        MODEL_BASE_PATH = properties['model_id']
    logging.info(f"Loading model in {MODEL_BASE_PATH}")

    #load hf model
    if not os.path.exists(MODEL_BASE_PATH):
        MODEL_BASE_PATH=snapshot_download(repo_id=properties['model_id']) 

    
    MODEL_BASE_PATH=MODEL_BASE_PATH+'/'
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
            MODEL_BASE_PATH+"mistral_7b_v0.2",
            map_location=device,
            amp='bf16',
            batch_size=1,
            tp_degree=2,
            n_positions=4096,
        )

        neuron_path=MODEL_BASE_PATH+'neuron_cache/bs1_tp2/'
        model_mistral.load(neuron_path)
        model_mistral.to_neuron()

    tokenizer_path= MODEL_BASE_PATH+"mistral_7b_v0.2" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    return model_mistral


def generate(inputs, byte_image):
    #image_files="https://llava-vl.github.io/static/images/view.jpg"

    t0=time.time()
    byte_data = base64.b64decode(byte_image)
    images = [Image.open(BytesIO(byte_data)).convert('RGB')]

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
        #logging.info(result)
        logging.info(f"handle: :Generation TIME TAKEN: { (time.time() - t1) * 1000} ms:::")
        return result


def handle(inputs: Input):
    #print('debug',inputs)
    try:
        global model_mistral
        global model_vit
        global model_projector
        global image_newline
        global embed_tokens
        global tokenizer
    
        if not model_mistral:
            model_mistral = load_model(inputs.get_properties())

        if inputs.is_empty():
            return None
        start_time = time.time()

        data = inputs.get_as_json()

        prompt = data["prompt"]
        byte_image = data["image"]
        params = data.get("parameters", None)

        result = generate(prompt, byte_image)
        '''
        logging.info(f"handle: :TIME:TAKEN:f{ (time.time() - start_time) * 1000}:ms:::")
        '''
        outputs= Output().add(result).add_property("content-type", "application/json")
    except:
        excep_str = traceback.format_exc()
        logging.info(f"error:in handle():: traceback={excep_str}:")
        outputs = Output().error(excep_str)

    return outputs
