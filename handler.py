""" Example handler file. """

import runpod
from PIL import Image
import trimesh
import torch
import uuid
import base64
from runpod.serverless.utils import rp_upload
import os
import sys
import tempfile
import boto3

from io import BytesIO
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
rembg = BackgroundRemover()
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path='tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo',
        use_safetensors=True,
        device = 'cuda',
)
pipeline.enable_flashvdm(mc_algo='mc')
pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
    model_path = 'tencent/Hunyuan3D-2',
)

def handler(job):
    """ Handler function that will be used to process jobs. """
    
    uid = uuid.uuid4()
    SAVE_DIR = 'gradio_cache'
    os.makedirs(SAVE_DIR, exist_ok=True)
    job_input = job['input']

    base_64_image = job_input.get('image', None)
    image = Image.open(BytesIO(base64.b64decode(base_64_image)))
    params = {}
    params['seed'] = job_input.get('seed', 1234)
    seed = params.get("seed", 1234)
    params['image'] = rembg(image)
    params['generator'] = torch.Generator('cuda').manual_seed(seed)
    params['octree_resolution'] = params.get("octree_resolution", 256)
    params['num_inference_steps'] = params.get("num_inference_steps", 5)
    params['guidance_scale'] = params.get('guidance_scale', 5.0)
    params['mc_algo'] = 'mc'

    
    import time
    start_time = time.time()
    mesh = pipeline(**params)[0]


    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
    mesh = pipeline_tex(mesh, image)


    type = params.get('type', 'glb')
    base64_output = ''

    s3 = boto3.client(
        service_name ="s3",
        region_name="auto", # Must be one of: wnam, enam, weur, eeur, apac, auto
        endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL", None),
        aws_access_key_id = os.environ.get("BUCKET_ACCESS_KEY", None),
        aws_secret_access_key = os.environ.get("BUCKET_ACCESS_SECRET", None)
    )
    with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            s3.upload_file(temp_file.name, os.environ.get("BUCKET_NAME", None),f'{uid}.{type}')
    
    torch.cuda.empty_cache()
    return f'{uid}.{type}'


runpod.serverless.start({"handler": handler})