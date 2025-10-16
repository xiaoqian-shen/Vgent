from transformers import AutoProcessor
import torch
from models.utils import fetch_video, resize_video
import numpy as np

def load_video(video_path, args):
    raw_video, frame_idx, fps = fetch_video({"video": video_path, "fps": args.fps}, resize=False)
    video, fps = resize_video(raw_video, fps, total_pixels=args.total_pixels*max(1, int(round(np.ceil(len(raw_video) / args.chunk_size))))*28*28)
    return [raw_video], None, None, frame_idx, fps, [video], None

def load_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    if "Qwen2.5" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        video_llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        video_llm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    video_llm.to("cuda")
    return None, video_llm, processor, None

def mllm_response(video_llm, tokenizer, processor, text, image_inputs, video, max_new_tokens=512, size_list=None, fps=None):
    if video is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "test.mp4",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    outputs = video_llm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_logits=True)
    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text