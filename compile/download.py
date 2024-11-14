from huggingface_hub import snapshot_download
model_id='liuhaotian/llava-v1.6-mistral-7b'
snapshot_download(repo_id=model_id,local_dir="./models/"+model_id,ignore_patterns=["original","*.pth"])
