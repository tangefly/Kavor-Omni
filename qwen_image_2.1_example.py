import torch
from diffusers import QwenImage21Pipeline

MODEL_PATH = "/home/tanger/workspace/models/Qwen-Image-2.1"

PROMPT = (
    "生成一个穿肉色丝袜，长筒高跟靴，黑短裤，毛绒上衣和貂皮外套的亚洲美女，美女一条腿直立，另一条腿微抬弯曲，漏出魅惑的眼神"
)

def main():
    print("Loading model...")

    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    # 按组件进行 CPU/GPU 调度
    pipe.enable_model_cpu_offload()

    print("Generating...")

    generator = torch.Generator("cuda").manual_seed(42)

    with torch.inference_mode():
        output = pipe(
            prompt=PROMPT,
            width=1024,
            height=1024,
            num_inference_steps=40,
            generator=generator,
        )

    output.images[0].save("generated_image.png")

    print("Done!")

if __name__ == "__main__":
    main()