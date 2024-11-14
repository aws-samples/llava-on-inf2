import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import MistralForCausalLM
import os

model_path="/home/ubuntu/llava16/LLaVA/models/liuhaotian/llava-v1.6-mistral-7b"
model_base=None
query="What are the things I should be cautious about when I visit here?"
image_files="https://llava-vl.github.io/static/images/view.jpg"
temperature=0.2
top_p=0.1
max_new_tokens=256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = MistralForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

saved_dir="./neuron/mistral-7b/"
model.save_pretrained(saved_dir)

tokenizer.save_pretrained(saved_dir)

model = LlavaMistralForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
)

print("start compiling......")

# compile
import torch_neuronx
vision_tower = model.get_vision_tower()
vision_tower.vision_tower.eval()
inputs = torch.rand([5, 3, 336, 336])
with torch.no_grad():
    vision_tower.vision_tower(inputs)
    model_neuronx = torch_neuronx.trace(vision_tower.vision_tower, inputs)
model_neuronx.save("./neuron/neuron-model-clip-vit.pt")


import torch_neuronx
inputs = torch.rand([5, 576, 1024], dtype=torch.float32)

with torch.no_grad():
    #y=model(inputs)
    #print(y.shape)
    model_neuronx = torch_neuronx.trace(model.model.mm_projector, inputs)

model_neuronx.save("./neuron/neuron-model-mm-projector.pt")

torch.save(model.model.image_newline, './neuron/image_newline.bin')


if not torch.cuda.is_available():
    print("load mistral........")
    from transformers_neuronx import MistralForSampling, GQA, NeuronConfig, QuantizationConfig, constants
    model_mistral = MistralForSampling.from_pretrained(
        model_path,
        map_location=torch.device('cpu'),
        amp='bf16',
        batch_size=1,
        tp_degree=2,
        n_positions=4096,
        #neuron_config=NeuronConfig(
        #    attention_layout=constants.LAYOUT_BSH,
        #    collectives_layout=constants.LAYOUT_BSH
        #    )
    )

    neuron_path='./neuron/bs1_tp2/'
    model_mistral.to_neuron()
    model_mistral.save(neuron_path)

torch.save(model.model.embed_tokens.state_dict(), './neuron/embed_tokens.bin')
