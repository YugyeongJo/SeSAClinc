# SeSAClinic 💆✨

> AI 기반 피부 분석 및 개인 맞춤형 케어 상담 챗봇 서비스 프로젝트

<p align="center">
  <img src="./img/webimg.png" width="32%" />
  <img src="./img/webimg2.png" width="32%" />
  <img src="./img/webimg3.png" width="32%" />
</p>

## 📖 Description
AI 기반 피부 분석 및 개인 맞춤형 케어 상담 챗봇 서비스는 AI 컴퓨터 비전 기술을 활용하여 사용자가 자신의 피부 상태를 진단받고, RAG(조회 기반 생성) 기반 상담 챗봇을 통해 개인 맞춤형 피부 고민 및 관리 방법을 상담받을 수 있는 서비스입니다. 

사용자가 얼굴 이미지를 업로드하면 AI가 피부 타입과 결함(홍조, 모공, 여드름, 색소침착, 건선 등)을 분석하고, 이를 바탕으로 간단한 관리 방법을 안내합니다.

보다 심도 있는 상담을 원하는 사용자는 RAG 기반의 챗봇을 통해 구체적인 피부 고민과 관리법에 대해 질문할 수 있습니다.  챗봇은 사용자의 개별적인 고민에 맞춘 맞춤형 응답을 제공하며, 보다 전문적인 피부 관리 정보를 전달합니다.

이 서비스는 사용자가 쉽고 간편하게 개인 맞춤형 피부 관리 방법을 찾을 수 있도록 돕고, 각자의 피부 고민에 대한 해결책을 제시하는 것을 목표로 합니다.

## 🗄️ Dataset
👉 Skin Type
- Total : 8410

|Skin Type|count|
|--|--|
|Dry|1906|
|Normal|3830|
|Oily|2674|

👉 Skin Defect

👉 LLM Document

## 🔧 Stack

### Language & Framework
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB"> <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi">

### DeepLearning
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"> <img src="https://img.shields.io/badge/torchvision-black.svg?style=for-the-badge&logo=torchvision&logoColor=white"> <img src="https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green"> 

### LLM
<img src="https://img.shields.io/badge/Openai-74aa9c?style=for-the-badge&logo=openai&logoColor=whit"> <img src="https://img.shields.io/badge/HuggingFace-%23FFBF00.svg?style=for-the-badge&logo=huggingface&logoColor=black"> <img src="https://img.shields.io/badge/glob-black.svg?style=for-the-badge&logo=&logoColor=white"> <img src="https://img.shields.io/badge/faiss-black.svg?style=for-the-badge&logo=&logoColor=white"> <img src="https://img.shields.io/badge/pdfplumber-black.svg?style=for-the-badge&logo=&logoColor=white">

### Data Handling
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/pillow-black.svg?style=for-the-badge&logo=&logoColor=white"> <img src="https://img.shields.io/badge/imgaug-black.svg?style=for-the-badge&logo=&logoColor=white"> <img src="https://img.shields.io/badge/matplotlib-3776AB.svg?style=for-the-badge&logo=&logoColor=white"> <img src="https://img.shields.io/badge/beautifulsoup4-3776AB.svg?style=for-the-badge&logo=beautifulsoup4&logoColor=white">

### Environment & Resource Management
<img src="https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white"> <img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">


## 📂 Directory Structure

```markdown
SeSAClinic/
├── data_preprocessing/
│      ├── data_labelling/
│      │      ├── COCOlabeling/
│      │      │        └── coco.py
│      │      ├── YOLOlabelling/
│      │      │        └── yolo.py
│      │      ├── ybat-master/
│      │      ├── dataset_check.py
│      │      ├── annotation_check.py
│      │      └── data_labelling_modify.py
│      ├── none_check.py
│      ├── data_sampling.py
│      └── data_split.py
├── computervision_modeling/
│      ├── Image_classification/
│      │      ├── Alexnet/
│      │      │        ├── data_augmentation.py
│      │      │        ├── alexnet.py
│      │      │        ├── flushing.py
│      │      │        └── wrinkle.py
│      │      └── VGG/
│      │           ├── skintype_vgg16_final.py
│      │           └── pores_vgg16_final.py
│      └── Object_detection/
│                 ├── FasterRCNN/
│                 │       ├── data.py
│                 │       ├── fasterrcnn_model.py
│                 │       ├── pre_train_evaluation.py
│                 │       ├── train_evaluation.py
│                 │       ├── match_label.py
│                 │       ├── matrix_map.py
│                 │       └── main.py
│                 └── YOLO/
├── llm/
│     ├── embedding/
│     │        ├── llm_embedding_KoBERT.py
│     │        ├── llm_embedding_KoELECTRA.py
│     │        ├── llm_embedding_list_kobert_4547.txt
│     │        └── llm_embedding_list_koelectra_3208.txt
│     ├── vectior_database/
│     │        ├── faiss_index_file_kobert_4547.index
│     │        ├── faiss_index_file_koelectra_3208.index
│     │        └── guide.xlsx
│     ├── document_parser.py
│     ├── summary_with_gpt.py
│     └── RAG-query_with_gpt.py
└── img/
│     ├── webimg.png
│     ├── webimg2.png
│     └── webimg3.png
└── icons/
```

## 💡 Team Members 
|이름|역할|Github|
|--|--|--|
|**조유경**|PM, Object Detecting Modeling, RAG LLM Engineer|https://github.com/YugyeongJo|
|**김태진**|Image classification Modeling, Web, MLops|https://github.com/dnwlwlq123|
|**한동우**|Object Detecting Modeling, RAG LLM Engineer|https://github.com/DongwooHan-GitHub|
|**박소연**|Image classification Modeling|https://github.com/amnyday|