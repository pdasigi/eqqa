local transformer_model = "roberta-base";
local epochs = 10;
local batch_size = 4;
local num_gradient_accumulation_steps = 2;

local train_data_path = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/metric-modeling/preproc/train_split_test_015_1231331.json";
local dev_data_path = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/metric-modeling/preproc/dev_split_test_015_1231331";

local training_data_size = 512;
local num_gpus = 1;

local target_metrics = ["meteor", "rougeL", "bleurt", "precision", "recall"];
local target_correctness = "human_correctness";

{
    "dataset_reader": {
        "type": "mocha_eqqa",
        "transformer_model_name": transformer_model,
        "target_correctness": target_correctness,
        "target_metrics": target_metrics,
        "target_datasets": ["drop"],
        "max_answer_length": 100,
        "max_query_length": 100,
        "max_document_length": 512,
        "include_context": true,
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty"
    },
    "model": {
        "type": "mocha_eqqa_roberta",
        "hidden_size": 256,
        "train_base": true,
        "target_metrics": target_metrics,
        "target_correctness": target_correctness,
        "encoder_output_projector": {
          "input_dim": 768,
          "num_layers": 2,
          "hidden_dims": [384, 1],
          "activations": "tanh",
          "dropout": 0.1
	      },
        "decoder_output_projector": {
          "input_dim": 1,
          "num_layers": 2,
          "hidden_dims": [16, 32],
          "activations": "tanh",
          "dropout": 0.1
        }
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
      "use_amp": true,
      "cuda_device": 0
    },
    "pytorch_seed": 4849,
}
