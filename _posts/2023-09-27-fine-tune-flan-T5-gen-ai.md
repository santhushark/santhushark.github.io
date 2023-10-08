---
layout: post
title: "Fine-Tune a FLAN-T5 Gen-AI Model"
description: "Fine-Tune an existing Large Language Model FLAN-T5 from Hugging Face for Enhanced Dialogue Summarization"
author: santhushark
category: Generative AI / LLM / NLP
tags: tansformers deep-learning LLM nlp generative-ai peft Hugging-face
finished: false
---
## Introduction


What is a FLAN-T5?

FLAN-T5 is a very good general purpose Large Language Model that is capable of doing a whole lot of things. It is an enhance version of T5 that has been finetuned in a mixture of tasks. 

What is LLM Fine-Tuning?

Fine tuning is a supervised learning process where you use a data set of labelled examples to update the weights of the Large Language Model. The labelled examples are prompt completion pairs, the fine-tuning process extends the trainging of the model to improve its ability to generate good completions for a specific task.

One strategy, known as instruction fine-tuning is particularly good at improving a model's performance on a variety og tasks. Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. The process results in a new version of the model with updated weights.

The output of an LLM is a probability distribution across tokens. So you compare the distribution of the completion and that of the training label and use the standard cross-entropy function to calculate the loss between the two token distributions. And then use the calculated loss to update your model weights in standard back propagation.

## Data


You are going to continue experimenting with the DialogSum Hugging Face dataset. DialogSum is a large-scale dialogue summarization dataset, consisting of 13,460 (Plus 100 holdout data for topic generation) dialogues with corresponding manually labeled summaries and topics.

## FULL FINE-TUNING


#### Preprocess the Dialog-Summary Dataset

You need to convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM. Prepend an instruction to the start of the dialog with **Summarize the following conversation** and to the start of the summary with **Summary** as follows:

**Training prompt (dialogue):**

**Summarize the following conversation.**

   **Chris: This is his part of the conversation.**
   **Antje: This is her part of the conversation.**

**Summary:**

**Training response (summary):**

**Both Chris and Antje participated in the conversation.**

Then preprocess the prompt-response dataset into tokens and pull out their **input_ids**.


#### Fine-Tune the model with the Preprocessed Dataset

Utilize the built-in Hugging Face **Trainer** class. Pass the preprocessed dataset with reference to the original model. Other training parameters are found experimentally. Training a fully fine-tuned version of the model would take a few hours on a GPU.

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train()
```

#### Evaluate the Model Qualitatively
As with many GenAI applications, a qualitative approach where you ask yourself the question "Is my model behaving the way it is supposed to?" is usually a good starting point. In the example below, you can see how the fine-tuned instruct model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

```console
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
ORIGINAL MODEL:
#Person1#: You'd like to upgrade your computer. #Person2: You'd like to upgrade your computer.
---------------------------------------------------------------------------------------------------
Fine Tuned INSTRUCT MODEL:
#Person1# suggests #Person2# upgrading #Person2#'s system, hardware, and CD-ROM drive. #Person2# thinks it's great.
```

#### Evaluate the Model Quantitatively
The ROUGE metric helps quantify the validity of summarizations produced by models. It compares summarizations to a "baseline" summary which is usually created by a human. While not perfect, it does indicate the overall increase in summarization effectiveness that we have accomplished by fine-tuning. The results show substantial improvement in all ROUGE metrics:
```console
ORIGINAL MODEL:
{'rouge1': 0.24223171760013867, 'rouge2': 0.10614243734192583, 'rougeL': 0.21380459196706333, 'rougeLsum': 0.21740921541379205}
INSTRUCT MODEL:
{'rouge1': 0.41026607717457186, 'rouge2': 0.17840645241958838, 'rougeL': 0.2977022096267017, 'rougeLsum': 0.2987374187518165}
```

Absolute percentage improvement of INSTRUCT MODEL over ORIGINAL MODEL
```console
rouge1: 18.82%
rouge2: 10.43%
rougeL: 13.70%
rougeLsum: 13.69%
```

## Parameter Efficient Fine-Tuning (PEFT)
PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning - with comparable evaluation results as you will see soon.

PEFT is a generic term that includes Low-Rank Adaptation (LoRA) and prompt tuning (which is NOT THE SAME as prompt engineering!). In most cases, when someone says PEFT, they typically mean LoRA. LoRA, at a very high level, allows the user to fine-tune their model using fewer compute resources (in some cases, a single GPU). After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).

That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request. The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.

#### Setup the PEFT/LoRA model for Fine-Tuning
You need to set up the PEFT/LoRA model for fine-tuning with a new layer/parameter adapter. Using PEFT/LoRA, you are freezing the underlying LLM and only training the adapter. Have a look at the LoRA configuration below. Note the rank (r) hyper-parameter, which defines the rank/dimension of the adapter to be trained.

Let's do all the necessary imports:

```python
from peft import LoraConfig, get_peft_model, TaskType

#Add LoRA adapter layers/parameters to the original LLM to be trained.
lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)

#Define training arguments and create Trainer instance.
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()
```
#### Evaluate the Model Qualitatively
As with many GenAI applications, a qualitative approach where you ask yourself the question "Is my model behaving the way it is supposed to?" is usually a good starting point. In the example below, you can see how the fine-tuned instruct model is able to create a reasonable summary of the dialogue compared to the original inability to understand what is being asked of the model.

```console
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
ORIGINAL MODEL:
#Pork1: Have you considered upgrading your system? #Person1: Yes, but I'd like to make some improvements. #Pork1: I'd like to make a painting program. #Person1: I'd like to make a flyer. #Pork2: I'd like to make banners. #Person1: I'd like to make a computer graphics program. #Person2: I'd like to make a computer graphics program. #Person1: I'd like to make a computer graphics program. #Person2: Is there anything else you'd like to do? #Person1: I'd like to make a computer graphics program. #Person2: Is there anything else you need? #Person1: I'd like to make a computer graphics program. #Person2: I'
---------------------------------------------------------------------------------------------------
INSTRUCT MODEL:
#Person1# suggests #Person2# upgrading #Person2#'s system, hardware, and CD-ROM drive. #Person2# thinks it's great.
---------------------------------------------------------------------------------------------------
PEFT MODEL: #Person1# recommends adding a painting program to #Person2#'s software and upgrading hardware. #Person2# also wants to upgrade the hardware because it's outdated now.
```

#### Evaluate the Model Quantitatively

```console
ORIGINAL MODEL:
{'rouge1': 0.2334158581572823, 'rouge2': 0.07603964187010573, 'rougeL': 0.20145520923859048, 'rougeLsum': 0.20145899339006135}
INSTRUCT MODEL:
{'rouge1': 0.42161291557556113, 'rouge2': 0.18035380596301792, 'rougeL': 0.3384439349963909, 'rougeLsum': 0.33835653595561666}
PEFT MODEL:
{'rouge1': 0.40810631575616746, 'rouge2': 0.1633255794568712, 'rougeL': 0.32507074586565354, 'rougeLsum': 0.3248950182867091}
```
The results show less of an improvement over full fine-tuning, but the benefits of PEFT typically outweigh the slightly-lower performance metrics.

Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
```console
rouge1: 17.47%
rouge2: 8.73%
rougeL: 12.36%
rougeLsum: 12.34%
```

Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
```console
rouge1: -1.35%
rouge2: -1.70%
rougeL: -1.34%
rougeLsum: -1.35%
```
Here you see a small percentage decrease in the ROUGE metrics vs. full fine-tuned. However, the training requires much less computing and memory resources (often just a single GPU).