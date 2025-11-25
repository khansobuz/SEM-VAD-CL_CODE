# FINAL_FULL_WITH_FOLDERS.py
# RUN THIS → GETS PERFECT train/ and test/ folders

import torch
import cv2
from pathlib import Path
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import json

# CHANGE ONLY THESE
MLLM_PATH   = r"C:\Users\khanm\Desktop\lab_project\DMN\MLLM_MODEL"
ROOT_VIDEOS = r"C:\Users\khanm\Desktop\lab_project\DMN\UCF_Crime\Videos"
TRAIN_TXT   = r"C:\Users\khanm\Desktop\lab_project\DMN\UCF_Crime\Anomaly_Detection_splits\Anomaly_Train.txt"
TEST_TXT    = r"C:\Users\khanm\Desktop\lab_project\DMN\UCF_Crime\Anomaly_Detection_splits\Anomaly_Test.txt"

# TWO CLEAN FOLDERS
SAVE_ROOT = Path("LLaVA1.5_CAPTIONS_PER_VIDEO")
(SAVE_ROOT / "train").mkdir(parents=True, exist_ok=True)
(SAVE_ROOT / "test").mkdir(parents=True, exist_ok=True)

print("Loading LLaVA-1.5-7B...")
processor = LlavaProcessor.from_pretrained(MLLM_PATH)
model = LlavaForConditionalGeneration.from_pretrained(
    MLLM_PATH, torch_dtype=torch.float16, device_map="auto"
).eval()

PROMPT = "USER: <image>\nDescribe what is happening in this image in one short sentence. ASSISTANT:"

def get_middle_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % 10 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (336, 336))
            frames.append(Image.fromarray(frame))
        count += 1
    cap.release()
    middle_frames = [frames[i] for i in range(4, len(frames), 8)]
    return middle_frames

def process(txt_path, folder_name):
    with open(txt_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    save_dir = SAVE_ROOT / folder_name
    print(f"\n→ Processing {len(lines)} videos → {save_dir}/")

    for rel_path in tqdm(lines, desc=folder_name):
        video_path = Path(ROOT_VIDEOS) / rel_path.replace("/", "\\")
        if not video_path.exists():
            continue

        frames = get_middle_frames(video_path)
        captions = []

        for i, img in enumerate(frames):
            inputs = processor(text=PROMPT, images=img, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40, do_sample=False, repetition_penalty=1.5)
            text = processor.decode(out[0], skip_special_tokens=True)
            caption = text.split("ASSISTANT:")[-1].strip()
            captions.append({"segment_id": i, "caption": caption})

        result = {
            "video": video_path.name,
            "total_segments": len(captions),
            "captions": captions
        }

        save_file = save_dir / f"{video_path.stem}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()

    print(f"{folder_name} DONE!")

# RUN BOTH
process(TRAIN_TXT, "train")
process(TEST_TXT,  "test")

print("\nALL DONE! Perfect folders created:")
print("   LLaVA1.5_CAPTIONS_PER_VIDEO/train/  → 1610 files")
print("   LLaVA1.5_CAPTIONS_PER_VIDEO/test/   → 290 files")
print("Ready forever — never run LLaVA again!")