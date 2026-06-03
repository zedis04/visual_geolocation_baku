import argparse
import os
import numpy as np
from datasets import load_dataset
from datasets import Image
from datasets import ClassLabel
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor, CLIPForImageClassification
import evaluate
import torch

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

model_checkpoints = ["facebook/convnext-tiny-224", "geolocal/StreetCLIP", "microsoft/swin-tiny-patch4-window7-224","microsoft/resnet-50"]
SEED = 42
accuracy_metric = evaluate.load("accuracy")

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)
    top1 = accuracy_metric.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    top5 = np.mean([
        labels[i] in top5_preds[i]
        for i in range(len(labels))
    ])

    return {
        "accuracy": top1,
        "top5_accuracy": top5,
    }

def load_data(data_files):
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.cast_column("image", Image())

    dataset = dataset.map(lambda x: {"cell_id": str(x["cell_id"])})
    train_cells = sorted(set(dataset["train"]["cell_id"]))
    label2id = {str(cell): i for i, cell in enumerate(train_cells)}
    id2label = {i: str(cell) for cell, i in label2id.items()}

    def encode_label(example):
        label = example["cell_id"]
        example["label"] = label2id[str(label)]
        return example

    dataset = dataset.map(encode_label)
    class_label = ClassLabel(
    num_classes=len(id2label),
    names=[id2label[i] for i in range(len(id2label))]
    )
    dataset = dataset.cast_column("label", class_label)
    return dataset, label2id, id2label

def preprocess_data(dataset, image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch
    
    splits = dataset["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    )
    train_ds = splits["train"]
    val_ds = splits["test"]
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    return train_ds, val_ds

def train(model_checkpoint, image_processor, train_ds, val_ds, num_labels, epochs, batch_size, learning_rate, label2id, id2label, is_clip=False):
    if is_clip:
        model = CLIPForImageClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True, #because the pretrained model may have a different number of labels than our dataset
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, #because the pretrained model may have a different number of labels than our dataset
        )

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-mapillary-baku",
        remove_unused_columns=False,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size= batch_size,
        num_train_epochs= epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    image_processor.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def main(args):
    model_checkpoint = model_checkpoints[args.model]
    if model_checkpoint == "geolocal/StreetCLIP":
        is_clip = True
    else:
        is_clip = False
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    data_path = args.data
    print(f"Training {model_checkpoint} for {epochs} epochs with batch size {batch_size} and learning rate {lr}.")
    data_files = {
        "train": os.path.join(data_path, "train/mapillary_images_metadata_train.csv"),
    }

    dataset, label2id, id2label = load_data(data_files)
    print(f"Loaded dataset with {len(dataset['train'])} training examples.")
    num_classes = len(label2id)
    print(f"Number of classes: {num_classes}")

    image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
    train_ds, val_ds = preprocess_data(dataset, image_processor)
    print(f"Preprocessed dataset.")
    train(model_checkpoint, image_processor, train_ds, val_ds, num_classes,epochs, batch_size, lr, label2id, id2label, is_clip)
    print(f"Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model", type=int, default=1, help="ID of model architecture to use.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--data", type=str, help="Path to the data folder.")
    args = parser.parse_args()
    main(args)