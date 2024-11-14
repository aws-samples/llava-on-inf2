# Comipile llava-v16-mistral-7b



## Getting started

This tutorial will introduce how to build llava-v16-mistral-7b on AWS Inferentia2.

## Environment 

- [ ] Neuron SDK 2.19.1
- [ ] Transformer 4.40.2
- [ ] transformers-neuronx https://github.com/cszhz/transformers-neuronx/tree/sdk219-embeding

## Setup
1. Launch an inf2.8xl instance with Ubuntu 22.04 and install Neuron Driver
2. Create virutal environment and install neuron cc and runtime packages
   ```
   python3 -m venv llava16_venv
   source llava16_venv/bin/activate
   python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
   pip install torch-neuronx==1.13.1.1.15.0
   python -m pip install neuronx-cc==2.14.227.0 torch-neuronx torchvision
   pip install transformers==4.40.2
   pip install git+https://github.com/cszhz/transformers-neuronx.git@sdk219-embeding
   ```
3. Code preparation
   ```
   mkdir llava16
   cd llava16
   git clone https://github.com/haotian-liu/LLaVA
   cd LLaVA
   # copy download.py compile.py run.py utils.py to LLaVA directory
   ``` 
4. Download model
   ```
   python download.py
   ```
5. Compile model
   ```
   #before compile need to do some changes
   
   python compile.py
   ```
6. Test compiled model
   ```
   python run.py
   ``` 

## License
For open source projects, say how it is licensed.


