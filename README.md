# CMP719 - Computer Vision Prooject: Breast Cancer Diagnosis from Mammographic Images

**Course**: CMP719 – Computer Vision  
**Instructor**: Prof. Dr. Nazlı İkizler Cinbiş  
**Term**: Spring 2025  
**Student**: Berkin Alkan

## Project Overview

The early and accurate diagnosis of breast cancer from mammographic images continues to pose a significant challenge, particularly under limited data conditions. This study compares the effectiveness of a custom convolutional neural network (CNN) with two widely used transfer learning models, **VGG16** and **ResNet50**. It also explores a hybrid model that integrates CNN with a **Vision Transformer (ViT)**, as well as a standalone **EfficientNetB0** architecture.

All experiments were conducted on the **MIAS dataset**, consisting of 322 mammogram images from 161 patients. A standardized preprocessing pipeline was applied across all models. The evaluation is based on accuracy, precision, recall, and F1-score metrics.

The goal is to identify the most effective architecture and quantify the diagnostic performance gains of a hybrid ViT-CNN configuration in a low-data setting.

## Dataset

- **MIAS (Mammographic Image Analysis Society)**  
  Public dataset containing labeled mammogram images.

## Models Evaluated

- DenseNet121 (baseline)
- ResNet50
- Modified DenseNet121
- ViT + CNN (hybrid)
- EfficientNetB0 ✅ (Best performing)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score (Weighted, Class-specific)

---

> This repository contains sample codes used for model training, evaluation, and analysis as submitted for the CMP719 final project.
