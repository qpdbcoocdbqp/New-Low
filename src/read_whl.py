import os

for pkg in  [
    'fastapi-packages',
    'grpc-packages', 
    'lancedb-packages',
    'torch-cu128-packages', 
    'torch-packages',
    'transformers-packages',
    'openai-packages',
    'llama-index-packages',
    'langchain-packages',
    'google-adk-packages',
    ]:
    print(pkg)
    os.listdir(pkg)
    print("="*30)
