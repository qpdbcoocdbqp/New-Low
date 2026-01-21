# New-Low
Records for frequent package and dependencies whl. Playing with [New Low](https://www.youtube.com/watch?v=A5GYOsKLp6o).

* **About New Low**

> New Low·Middle Class Rut
>
> No Name No Color

## Packages

### OpenAI API client

* dependencies whl: [readme](source/openai-packages/README.md)

```sh
uv run --with pip python -m pip download openai litellm cohere \
-d ./openai-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### Lancedb

* dependencies whl: [readme](source/lancedb-packages/README.md)

```sh
uv run --with pip python -m pip download lancedb pylance duckdb pandas \
-d ./lancedb-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### PyTorch

* dependencies whl: [readme](source/torch-packages/README.md)

```sh
uv run --with pip python -m pip download torch torchvision \
-d ./torch-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### PyTorch (CUDA 12.8)

* dependencies whl: [readme](source/torch-cu128-packages/README.md)

```sh
uv run --with pip python -m pip download torch torchvision \    
--extra-index-url https://download.pytorch.org/whl/cu128 \
-d ./torch-cu128-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### Transformers

* dependencies whl: [readme](source/transformers-packages/README.md)

```sh
uv run --with pip python -m pip download transformers sentence-transformers fastembed datasets \
-d ./transformers-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### FastAPI

* dependencies whl: [readme](source/fastapi-packages/README.md)

```sh
uv run --with pip python -m pip download fastapi fastmcp uvicorn \
-d ./fastapi-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### gRPC

* dependencies whl: [readme](source/grpc-packages/README.md)

```sh
uv run --with pip python -m pip download grpcio grpcio-tools \
-d ./grpc-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### Llama Index

* dependencies whl: [readme](source/llama-index-packages/README.md)

```sh
uv run --with pip python -m pip download llama-index \
-d ./llama-index-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### LangChain

* dependencies whl: [readme](source/langchain-packages/README.md)

```sh
uv run --with pip python -m pip download langchain langgraph \
-d ./langchain-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### Google ADK

* dependencies whl: [readme](source/google-adk-packages/README.md)

```sh
uv run --with pip python -m pip download google-adk \
-d ./google-adk-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```

### Jupyterlab

* dependencies whl: [readme](source/jupyterlab-packages/README.md)

```sh
uv run --with pip python -m pip download jupyterlab \
-d ./jupyterlab-packages \
--platform win_amd64 \
--python-version 3.13 \
--abi cp313 \
--only-binary=:all:
```
