import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from transformers import StoppingCriteria
from decord import cpu, VideoReader
from PIL import Image
import time

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
    video = []
    for frame_index in frame_indices:
        img = vr[frame_index].asnumpy()
        video.append(img)
    raw_video = np.stack(video)
    image_sizes = [raw_video[0].shape[:2]]
    video = process_images(raw_video, image_processor, model.config)
    video = [item.unsqueeze(0) for item in video]
    return raw_video, None, None, frame_indices, fps, video, image_sizes

def load_model():
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        "Vision-CAIR/LongVU_Qwen2_7B", None, "cambrian_qwen",
    )
    model.cuda()
    model.eval()
    return tokenizer, model, image_processor, context_len

def mllm_response(video_llm, tokenizer, processor, text, image_inputs, video, max_new_tokens=512, size_list=None, fps=None):
    qs = DEFAULT_IMAGE_TOKEN + "\n" + text
    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(video_llm.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = video_llm.generate(
            input_ids,
            images=video,
            image_sizes=size_list,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()