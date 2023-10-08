---
layout: post
title: META SAM model workflow in GCP
description: "Deploy SAM on Google Cloud Platform using VertexAI for online Predictions. Build Batch Prediction pipelines using Kubeflow."
author: santhushark
category: Machine Learning Operations
tags: mlops computer-vision SAM
finished: true
---
## Introduction


#### What is Image Segmentation?
The process of partitioning an image into individual objects within an image is called Image Segmentation.

[<img src="/assets/img/gcp_sam/fish.gif" height="210" width="300"/>](/assets/img/gcp_sam/fish.gif?raw=true)

#### What is SAM?
**Segment Anything Model (SAM)** is a result of META's research in the Computer vision space. Trained on 11 million images and 1.1 billion masks, The model is capable of producing high-quality masks of objects within an image.

#### Can Segment Anything Model be a corner piece of any Computer Vision Framework?
Object detection is a common computer vision problem, Segmenting images to detect objects of interest can be a time consuming and labour intensive task. We can use computer vision frameworks to make this process more efficient and cost effective, and one model that’s recently been developed to help with segmentation tasks is the Segment Anything Model, which we’ll refer to here as SAM.

## Cloud Architecture leveraged to unlock scaling and segmenting thousands of images!


**Google Cloud Platform** is the cloud infrastructure we are using in this project. **VertexAI, Cloud Storage, Artifacts Registry** are some of the services utilised in the project.
![SAM-model-workflow](/assets/img/gcp_sam/sam_workflow.png?raw=true)

In the above Machine Learning workflow diagram we have two architectures:

#### 1)Online Predictions
+ This involves deploying the model to a REST endpoint. However it is not quite straight forward.
+ Here we will be storing the model that we will be using for deployment in Cloud Storage.
+ SAM is a custom model, Getting predictions out from SAM involves an image going through custom data pre-processing followed by prediction and post-processing.
+ The Custom prediction routine is capable of handling model input that constitutes image base64 string with/without prompts.
+ Hence deploying it on VertexAI requires building a custom container with custom prediction routine.
+ The built container is pushed to a Repository in Artifacts Registry and later used in the Model Registry Step.
+ Once the model is registered, The model is deployed to end-point using Online Predictions tool within VertexAI.
+ The Response JSON structure of the endpoint is of the format:
  - Predicting with prompts
  ```python
  {
    "file_path"                 : file name,
    "masks"                     : [masks],
    "scores"                    : [scores],
    "logits"                    : [logits]
  }
  ```
  
  - Predicting without prompts
  ```python
  {
    "file_path"                 : file name,
    "masks"                     : [masks]
  }
  ```
+ An example image segment from SAM endpoint, Here we are co-ordinate prompts to point the person in the image:
![endpoint-response](/assets/img/gcp_sam/club-house-segment-with-prompts.png?raw=true)
  

#### 2)Batch Prediction Pipeline
+ Here we leverage the VertexAI Pipeline tool and SAM model to segment images at scale.
+ The model that will be used in the pipeline to segment images will be stored in Cloud storage.
+ Kubeflow SDK is used to build pipelines.
+ The images that need to be processed is stored inside a folder in Cloud Storage.
+ The images are processed in sequence inside the pipeline and the predictions are output into a markdown for static visualization.
+ Below is an example of Static Visualization of a single image in a pipeline
  
![endpoint-response](/assets/img/gcp_sam/static_visualization.png?raw=true)

+ Similarly multiple pipeline experiments were conducted on different set of images.
+ The batch prediction pipeline experimental results are graphically represented as below:
  
![experiments_plot_1](/assets/img/gcp_sam/plot_1.png?raw=true)

![experiments_plot_2](/assets/img/gcp_sam/plot_2.png?raw=true)


 


