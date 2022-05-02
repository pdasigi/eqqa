local transformer_model = "roberta-base";
local epochs = 10;
local batch_size = 64;
local num_gradient_accumulation_steps = 1;

local train_data_path = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/mocha_eqqa_data/split__train_candidates_only.json";
local dev_data_path = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/mocha_eqqa_data/split__dev_candidates_only.json";

local training_data_size = 512;
local num_gpus = 1;

{
    "dataset_reader": {
        "type": "mocha_eqqa",
        "transformer_model_name": transformer_model,
        "max_answer_length": 100,
        "max_query_length": 100,
        "max_document_length": 512,
        "include_context": true
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty"
    },
    "model": {
        "type": "mocha_eqqa_roberta",
        "hidden_size": 256,
        "train_base": true
    },
    "data_loader": {
        "batch_size": batch_size
    },
    "trainer": {
      "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.0,
        "lr": 5e-4,
        "eps": 1e-8
      },
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": epochs,
        "cut_frac": 0.1,
        "num_steps_per_epoch": std.ceil(training_data_size / (batch_size * num_gradient_accumulation_steps * num_gpus))
      },
      "callbacks": [
	{"type": "tensorboard"}
      ],
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
      "patience": epochs,
      "enable_default_callbacks": false,
      "use_amp": false,
      "cuda_device": 0
    },
    "pytorch_seed": 15371,
}
