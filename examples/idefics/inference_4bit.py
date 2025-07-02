# 이것은 4비트 양자화를 사용하는 IDEFICS-9B 추론 데모이며 약 7GB의 GPU 메모리가 필요합니다.
# 이를 통해 Google Colab에서도 실행할 수 있습니다.

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b"
#checkpoint = "HuggingFaceM4/tiny-random-idefics"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
processor = AutoProcessor.from_pretrained(checkpoint)

prompts = [
    "Instruction: provide an answer to the question. Use the image to answer.\n",
    "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg",
    "Question: What's on the picture? Answer: \n"
]

inputs = processor(prompts, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=150)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])
