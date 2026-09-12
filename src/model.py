import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_CONFIG = dict(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)


def load_full_ft_model(id2label, label2id, num_labels=4):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype="auto",
        device_map="auto",
    )
    return model


def load_lora_model(id2label, label2id, num_labels=4):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype="auto",
        device_map="auto",
    )
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.enable_input_require_grads()
    return model


def load_qlora_model(id2label, label2id, num_labels=4):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.enable_input_require_grads()
    return model


MODEL_LOADERS = {
    "full_ft": load_full_ft_model,
    "lora": load_lora_model,
    "qlora": load_qlora_model,
}


def load_model(method, id2label, label2id, num_labels=4):
    if method not in MODEL_LOADERS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(MODEL_LOADERS)}")
    return MODEL_LOADERS[method](id2label, label2id, num_labels)