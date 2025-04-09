
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    try:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path='tencent/Hunyuan3D-2mini',
                subfolder='hunyuan3d-dit-v2-mini-turbo',
                use_safetensors=True,
                device = 'cuda',
        )
    except:
        print("shit occurred")
    try:
        pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
            model_path = 'tencent/Hunyuan3D-2',
        )
    except:
        print("shit occurred")


if __name__ == "__main__":
    get_diffusion_pipelines()