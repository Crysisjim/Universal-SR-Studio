"""pip constraints = exact versions of the shared venv this release was tested with.

Generated 2026-11-26 from runtimes/.venv (torch 2.7.0+cu126). Passed with `-c` to every
dependency install, so a new user gets the same packages as the tested build instead of
whatever PyPI serves that day. torch/torchvision and their own dependencies are NOT listed:
they come from the CUDA wheel index chosen for the user's GPU (as do nvidia-* / triton).
"""

CONSTRAINTS = """\
absl-py==2.5.0
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.8.0
antialiased-cnns==0.3
anyio==4.14.2
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.4.2
colorama==0.4.6
einops==0.8.2
ema-pytorch==0.8.3
filelock==3.32.3
flatbuffers==25.12.19
fsspec==2026.7.0
grpcio==1.83.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.28.0
idna==3.19
Jinja2==3.1.6
lmdb==2.3.0
markdown-it-py==4.2.0
Markdown==3.10.3
MarkupSafe==3.0.3
mdurl==0.1.2
ml_dtypes==0.6.0
mpmath==1.3.0
msgspec==0.21.1
networkx==3.6.1
ninja==1.13.2
numpy==2.5.2
nvidia-ml-py==13.610.43
onnx==1.22.0
onnxruntime-gpu==1.26.0
opencv-python-headless==5.0.0.93
opencv-python==5.0.0.93
packaging==26.3
pillow==12.3.0
protobuf==7.36.0
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
Pygments==2.21.0
pytorch_optimizer==3.10.1
pyvips-binary==8.18.5
pyvips==3.1.1
PyWavelets==1.9.0
PyYAML==6.0.3
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scipy==1.18.1
simsimd==6.5.16
spandrel==0.4.2
spandrel_extra_arches==0.2.0
stringzilla==5.1.2
sympy==1.14.0
tensorboard-data-server==0.7.2
tensorboard==2.21.0
timm==1.0.28
toml==0.10.2
tqdm==4.70.0
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.7.0
Werkzeug==3.1.8
"""
