🧠 SEM-VAD-CL: Continual Learning for Weakly Supervised Video Anomaly Detection

This project implements a continual learning framework for Weakly Supervised Video Anomaly Detection (WSVAD). we propose SEM-VAD-CL, a novel framework that integrates semantic reasoning, explainability, and continual learning for WSVAD. The framework employs pre-extracted video features combined with a lightweight large language model (LLM) contextualizer to generate seman-
tic textual descriptions of video events, enabling anomaly scoring that is both interpretable and transparent. A Reasoned Anomaly Interpreter utilizes Chain-of-Thought (CoT)-style reasoning to provide human-readable justifications for anomaly predictions, while an Adaptive Continual Module incrementally updates knowledge of normal and abnormal patterns through temporal consistency and memory integration, mitigating catastrophic forgetting. By bridging traditional feature-driven detection with language-based reasoning and continual learning, SEM-VAD-CL offers a robust, efficient, and explainable solution for real-world anomaly detec-
tion. 

<img width="1578" height="587" alt="image" src="https://github.com/user-attachments/assets/47baa07f-1ca1-4028-b2be-5ab9056dd9d4" />



✅ Requirements

Install the required Python libraries:

pip install torch==2.4.1 torchvision==0.20.0 numpy==1.24.4 pandas==2.0.3 scikit-learn sentence-transformers


Recommended:

Python 3.8+

GPU with ≥16 GB memory

CUDA-compatible PyTorch

📁 Project Structure
SEM-VAD-CL/
│
├── dataset1.py/
│   └── Pre-extracted I3D features for UCF-Crime / ShanghaiTech
│
├── datasetPreprocessingMLLM_offline.py
│   └── Python script to generate offline MLLM captions using LLaVA-1.5
│
├── LLaVA1.5_CAPTIONS_PER_VIDEO/
│   ├── train/   → JSON captions for training videos
│   └── test/    → JSON captions for testing videos
│
├── main.py
│   └── Main pipeline using pretrained I3D features + SEM-VAD-CL
│
├── MLLM_Captions_SEM_VAD_CL.py
│   └── Evaluate directly using offline MLLM JSON captions
│
├── ProtoSEM-CL/
│   └── Lightweight prototype-based continual learning module
│
└── Sub_classes.py/
    └── Evaluate subclass AUC & fine-grained performance

📂 Dataset Structure
UCF-Crime/
├── all_rgbs/
├── all_flows/
├── train_normal.txt
├── train_anomaly.txt
├── test_normalv2.txt
├── test_anomalyv2.txt

Offline MLLM Captions
LLaVA1.5_CAPTIONS_PER_VIDEO/
├── train/*.json
└── test/*.json

Ensure `.npy` files for RGB and optical flow are stored under `all_rgbs/` and `all_flows/`.

Dataset link:
url: https://drive.google.com/file/d/1bOpDDDa0ZTyV0q9-V8HFEXCYPFeHlDcr/view?usp=sharing


🚀 How to Run main file

python main.py


🚀 How to Run other supporting file
Script / Folder	Purpose
datasetPreprocessingMLLM_offline.py	Generate offline MLLM JSON captions from videos
main.py	Train/evaluate SEM-VAD-CL using pretrained I3D features
MLLM_Captions_SEM_VAD_CL.py	Run inference directly with offline JSON captions
ProtoSEM-CL/	Prototype-based continual learning module
Sub_classes/	Evaluate fine-grained subclass AUC & performance
💾 Pretrained Model

Checkpoint file:

ckpt_auc_0.8589.pth


 

⚙️ System Requirements

GPU with 16 GB memory recommended

Python 3.8+

PyTorch 1.9+

📝 Author

Name: KHAN MD SABUJ
Email: khansobuz203@gmail.com
