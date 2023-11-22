import torch
import numpy as np

from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, TrainingArguments, Trainer
from torchvision.transforms import (Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    Resize,
                                    ToTensor,
                                    RandAugment)
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod, Options


def model_compression(num, compression_type="l2norm"):
    compressor = ModelCompressor(email=args.email, password=args.password)

    # Upload Model
    UPLOAD_MODEL_NAME = "vit"
    TASK = Task.IMAGE_CLASSIFICATION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = "model_vit.pt"
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [224, 224]}]

    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    if compression_type == "l2norm":
        COMPRESSED_MODEL_NAME = "l2norm_vit"
        COMPRESSION_METHOD = CompressionMethod.PR_L2
        RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
        RECOMMENDATION_RATIO = 0.2

        OUTPUT_PATH = f"compressed_vit_l2norm_{num}.pt"
        opt=Options(group_policy='none')
        compressed_model = compressor.recommendation_compression(
            model_id=model.model_id,
            model_name=COMPRESSED_MODEL_NAME,
            compression_method=COMPRESSION_METHOD,
            recommendation_method=RECOMMENDATION_METHOD,
            recommendation_ratio=RECOMMENDATION_RATIO,
            output_path=OUTPUT_PATH,
            options=opt,
        )
    elif "svd":
        COMPRESSED_MODEL_NAME = "fdsvd_vit"
        COMPRESSION_METHOD = CompressionMethod.FD_SVD
        RECOMMENDATION_METHOD = RecommendationMethod.VBMF
        RECOMMENDATION_RATIO = 0.2

        OUTPUT_PATH = f"compressed_vit_svd_{num}.pt"
        compressed_model = compressor.recommendation_compression(
            model_id=model.model_id,
            model_name=COMPRESSED_MODEL_NAME,
            compression_method=COMPRESSION_METHOD,
            recommendation_method=RECOMMENDATION_METHOD,
            recommendation_ratio=RECOMMENDATION_RATIO,
            output_path=OUTPUT_PATH,
        )
    return OUTPUT_PATH

def load_vit_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('model.pt', map_location=device)
    _model = torch.load(model_name, map_location=device)

    model.vit = _model

    return model

def load_datasets():
    # load dataset
    dataset = load_dataset('cifar100')
    image = dataset["train"]["img"]

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    # load feature extractor$
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    def transform(example_batch):
        def toRGB(image):
            if image.mode != 'RGB':
                return image.convert("RGB")
            return image
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([toRGB(x) for x in example_batch['img']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['label'] = example_batch['coarse_label']
        return inputs

    prepared_ds = dataset.with_transform(transform)

    labels = dataset['train'].features['coarse_label'].names

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    _train_transforms = Compose(
        [
            Resize([feature_extractor.size["width"], feature_extractor.size["height"]]),
            RandomHorizontalFlip(),
            RandAugment(),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        examples['label'] = examples['coarse_label']
        return examples
    
    # Set transforms
    train_ds.set_transform(train_transforms)
    _val_transforms = Compose(
            [
                Resize([feature_extractor.size["width"], feature_extractor.size["height"]]),
                ToTensor(),
                normalize,
            ]
        )
    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        examples['label'] = examples['coarse_label']
        return examples
    test_ds.set_transform(val_transforms)
    
    return train_ds, test_ds, prepared_ds

def params_comparision(original_model, compressed_model):
    from thop import profile
    from torchinfo import summary

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 이슈로 남기기
    input_size = 224

    dummy_input = torch.ones((1,3,input_size,input_size)).to(device)

    original_params = summary(original_model.vit.eval(), (1,3,input_size, input_size)).trainable_params
    compressed_params = summary(compressed_model.vit.eval(), (1,3,input_size, input_size)).trainable_params

    print("\n--- Original model ==> Compressed model ---")
    result = f"Params: {original_params} ==> {compressed_params} (x{round(original_params/compressed_params,3)})"
    print(result)
    return result

def get_model_accuracy(model, train_ds, test_ds, prepared_ds):
    
    training_args = TrainingArguments(
    output_dir=f"./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=0,
    fp16=False,
    save_steps=392,
    eval_steps=392,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    eval_accumulation_steps=30,
    )

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }

    metric = load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions[0], axis=1), references=p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
    )

    # predict 
    predictions = trainer.predict(prepared_ds['test'])
    preds = np.argmax(predictions.predictions[0], axis=-1)
    result = metric.compute(predictions=preds, references=predictions.label_ids)
    print(result)
    return result

def train_model(model, train_ds, test_ds, prepared_ds, seed):
    # training arguments
    training_args = TrainingArguments(
    output_dir=f"./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=1,
    fp16=False,
    save_steps=392,
    eval_steps=392,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    eval_accumulation_steps=30,
    seed=seed,
    data_seed=seed,
    )

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }
    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions[0], axis=1), references=p.label_ids)

    compressed_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=feature_extractor,
    )

    predictions = compressed_trainer.predict(prepared_ds['test'])
    preds = np.argmax(predictions.predictions[0], axis=-1)
    result = metric.compute(predictions=preds, references=predictions.label_ids)
    print(result)

    # training
    train_results = compressed_trainer.train()
    compressed_trainer.save_model()
    compressed_trainer.log_metrics("train", train_results.metrics)
    compressed_trainer.save_metrics("train", train_results.metrics)
    compressed_trainer.save_state()

    predictions = compressed_trainer.predict(prepared_ds['test'])
    preds = np.argmax(predictions.predictions[0], axis=-1)
    result = metric.compute(predictions=preds, references=predictions.label_ids)
    print(result)
    
    return result

def compare_accuracy(accu1, accu2):
    print(f"model accuracy: {accu1} =====> {accu2}")
    return

def main(args):

    # load dataset
    train_ds, test_ds, prepared_ds = load_datasets()

    compressed_model_path = model_compression(i, args.compression_type)

    # load model
    original_model = load_vit_model("model_vit.pt")
    compressed_model = load_vit_model(compressed_model_path)

    # compression relsult (# of param)
    comparision_result = params_comparision(original_model, compressed_model)
    
    # accuracy check (before fine-tuning)
    original_accuracy = get_model_accuracy(original_model, train_ds, test_ds, prepared_ds)
    compressed_model_before = get_model_accuracy(compressed_model, train_ds, test_ds, prepared_ds)
    compare_accuracy(original_accuracy, compressed_model_before)
    
    # fine-tuning
    compressed_model_after = train_model(compressed_model, train_ds, test_ds, prepared_ds, i)
    
    # accuracy check (after fine-tuning)
    compare_accuracy(original_accuracy, compressed_model_after)
    results = f"original_accuracy: {original_accuracy}, compressed_accuracy: {compressed_model_after}, {comparision_result}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('-mail', '--email', type=str, default="email", help='')
    parser.add_argument('-pw', '--password', type=str, default="password", help='')
    args = parser.parse_args()

    main(args)
