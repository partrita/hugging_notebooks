# https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/image_captioning.ipynb 에서 수정됨

# 이 예제는 일반적인 미세 조정(peft 없음)을 보여줍니다. 메모리 요구 사항을 작게 유지하기 위해
# 원본 사전 학습된 텍스트 및 이미지 레이어를 고정하여 메모리 요구 사항을 40GB로 유지합니다.
# 여러 GPU가 있는 경우 고정 해제 부분을 제거하여 전체 모델을 미세 조정할 수 있습니다.
# 또는 IDEFICS_finetuning_demo.ipynb 노트북에 표시된 PEFT 솔루션을 사용하면
# 전체 모델을 미세 조정하는 데 20GB만 필요합니다.

import torch
import torchvision.transforms as transforms

from datasets import load_dataset
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
# checkpoint = "HuggingFaceM4/tiny-random-idefics"

processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)

# 원본 텍스트 및 비전 모델을 고정하고 IDEFICS에서 추가한 레이어만 미세 조정합니다.
# 전체 모델을 고정 해제할 수 있지만 미세 조정하려면 여러 GPU가 필요합니다.
model.model.freeze_text_layers()
model.model.freeze_vision_layers()

# 도움말 유틸리티
def check_inference():
    url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/pokemon.png"
    prompts = [
        url,
        "Question: What's on the picture? Answer:",
    ]

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=150)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

# 미세 조정 전에 생성 확인
check_inference()
# 음, 실제로는 모델이 이미 포켓몬을 알고 있는 것 같습니다. 하지만 이 데이터세트는 이를 더욱 구체화할 것입니다.

# 포켓몬 유형 데이터세트에서 모델 미세 조정
ds = load_dataset("GabeHD/pokemon-type-captions")
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
eval_ds = ds["test"]

def convert_to_rgb(image):
    # `image.convert("RGB")`는 투명 이미지에 대해 잘못된 배경을 생성하므로 .jpg 이미지에만 작동합니다.
    # `alpha_composite` 호출은 이 경우를 처리합니다.
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def ds_transforms(example_batch):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch)):
        prompts.append(
            [
                example_batch["image"][i],
                f"Question: What's on the picture? Answer: {example_batch['text'][i]}\n",
            ],
        )

    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)

    inputs["labels"] = inputs["input_ids"]

    return inputs

train_ds.set_transform(ds_transforms)
eval_ds.set_transform(ds_transforms)

model_name = checkpoint.split("/")[1]

# 이 설정에는 약 40GB의 GPU 메모리가 필요합니다.
training_args = TrainingArguments(
    output_dir=f"{model_name}-pokemon",
    learning_rate=5e-6,
    num_train_epochs=10,
    bf16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    dataloader_pin_memory=False,
    save_total_limit=3,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=1000, # 준비될 때까지 저장하지 마세요...
    eval_steps=40,
    logging_steps=40,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

# 미세 조정 후 생성 다시 확인
check_inference()

# 미세 조정 후 이상적으로는 다음과 같은 것을 생성하기를 원합니다: 분홍색과 파란색 포켓몬 그림
