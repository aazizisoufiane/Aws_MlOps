import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from transformers import EvalPrediction
from transformers import EarlyStoppingCallback, IntervalStrategy
import torch


def main(args):
    # Hyper-parameters
    metric_name = "f1"
    training_dir = args.training_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir
    labels_dir = args.labels_dir
    # checkpoint_s3_uri = args.checkpoint_s3_uri

    model_name = args.model_name
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    eval_steps = args.eval_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate

    def preprocess_data(examples):
        # take a batch of texts
        text = examples["ABSTRACT"]
        # encode them
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]
        encoding["labels"] = labels_matrix.tolist()
        return encoding

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info(sys.argv)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    train_file = f"{training_dir}/train.csv"
    test_file = f"{test_dir}/test.csv"
    labels_file = f"{labels_dir}/labels.csv"
    logger.info(f"labels_file: {labels_file}")
    logger.info(f"test_file: {test_file}")
    logger.info(f"train_file: {train_file}")

    dataset = load_dataset('csv', data_files={'train': train_file,
                                              'test': test_file,
                                              #                                              'labels': labels_file
                                              })

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    labels_df = pd.read_csv(labels_file)  # load_from_disk(labels_file)
    logger.info(f" labels_df shape: {labels_df.shape}")
    #     logger.info(f"dataset_train columns: {train_dataset.columns}")
    dim_train = len(train_dataset)

    labels = labels_df.labels.values.tolist()
    num_labels = len(labels)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    logger.info(f" length labels is: {num_labels}")

    # tokenizer helper function
    def tokenize(batch):
        #         logger.info(f"tokenizer: {batch["ABSTRACT"]}")
        return tokenizer(batch["ABSTRACT"], padding='max_length', truncation=True)

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_data, batched=True, remove_columns=test_dataset.column_names)

    # set format for pytorch
    train_dataset.set_format('torch')

    test_dataset.set_format('torch')

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for classification

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        recall = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')

        try:
            roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        except:
            roc_auc = 0
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy,
                   'recall': recall,
                   'precision': precision}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    logger.info(f"length of id2label: {len(id2label)}")
    logger.info(f"length of label2id: {len(label2id)}")

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               problem_type="multi_label_classification",
                                                               num_labels=num_labels,
                                                               id2label=id2label,
                                                               label2id=label2id,
                                                               )
    logger.info(f"model:{model}")
    try:
        outputs = model(input_ids=train_dataset['input_ids'][0].unsqueeze(0),
                        labels=train_dataset['labels'][0].unsqueeze(0))
        logger.info(f"outputs: {outputs}")
    except Exception as e:
        logging.critical(f"outputs failled:{e}", exc_info=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=IntervalStrategy.STEPS,  # "epoch" ,
        save_strategy="steps",
        eval_steps=eval_steps,  # 50, # ADDED Evaluation and Save happens every 50 steps
        save_total_limit=5,  # ADDED Only last 5 models are saved. Older ones are deleted.

        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        warmup_steps=warmup_steps,
        logging_dir=f"{output_data_dir}/logs",
        push_to_hub=False,
    )

    logger.info(f"training_args:{training_args}")
    #     outputs = model(input_ids=train_dataset['input_ids'][0].unsqueeze(0), labels=train_dataset[0]['labels'].unsqueeze(0))
    early_stopping_patience = int(dim_train / train_batch_size)
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    )
    logger.info(f"trainer:{trainer}")
    logger.info(f"output_dir:{output_dir}")
    last_checkpoint = get_last_checkpoint(args.output_dir)

    logger.info(f"get_last_checkpoint:{last_checkpoint}")
    #     train model
    if get_last_checkpoint(output_dir) is not None:
        try:

            trainer.train(resume_from_checkpoint=last_checkpoint)
            logger.info("***** continue training *****")

        except:
            logging.info("Initialize Training")
            trainer.train()

    else:
        trainer.train()

    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    logger.info(f"Eval results: {eval_result}")

    logger.info(f"output_data_dir:{output_data_dir}")

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    logger.info(f"model_dir:{model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--output-dir", type=str, default='/opt/ml/checkpoints')
    # parser.add_argument("--output_dir", type=str,  default='/opt/ml/checkpoints')  
    # parser.add_argument("--checkpoint_s3_uri", type=str)

    parser.add_argument("--eval_steps", type=int, default=50)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--labels_dir", type=str, default=os.environ["SM_CHANNEL_LABELS"])

    args, _ = parser.parse_known_args()

    main(args)
