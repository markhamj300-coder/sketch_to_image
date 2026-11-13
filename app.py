import torch, numpy as np, gradio as gr
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN   = "lllyasviel/control_v11p_sd15_scribble"
SIZE = 512
NEG  = "lowres, blurry, deformed, bad anatomy, extra limbs, watermark, text, cropped"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(CN, torch_dtype=dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE, controlnet=controlnet, torch_dtype=dtype)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

def prepare_control_image(sketch, size=SIZE, threshold=170, dilate=3):
    g = ImageOps.autocontrast(sketch.convert("L"))
    g = ImageOps.pad(g, (size, size), color=255, method=Image.LANCZOS)
    bw = g.point(lambda v: 255 if v < threshold else 0)
    bw = bw.filter(ImageFilter.MaxFilter(dilate))
    return bw.convert("RGB")

def run(sketch, prompt, cn_scale, guidance, steps, seed):
    if sketch is None:
        return None, None
    if isinstance(sketch, np.ndarray):
        sketch = Image.fromarray(sketch)
    ctrl = prepare_control_image(sketch)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    out = pipe(prompt=(prompt or "a detailed, high quality image") + ", highly detailed",
               negative_prompt=NEG, image=ctrl, num_inference_steps=int(steps),
               guidance_scale=float(guidance), controlnet_conditioning_scale=float(cn_scale),
               generator=gen).images[0]
    return ctrl, out

demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(label="Sketch (dark lines on white)", type="pil", image_mode="L"),
            gr.Textbox(label="Prompt"),
            gr.Slider(0, 2, value=1.0, step=0.1, label="Sketch adherence"),
            gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale"),
            gr.Slider(10, 50, value=30, step=1, label="Steps"),
            gr.Number(value=42, precision=0, label="Seed")],
    outputs=[gr.Image(label="Control image"), gr.Image(label="Generated image")],
    title="Sketch → Image (SD1.5 + ControlNet Scribble)",
    examples=[["sketches/your_sketch.png", "a futuristic glass tower at sunset", 1.0, 7.5, 30, 42]],  # edit or remove
)

if __name__ == "__main__":
    demo.launch(share=True)
