local transformer_model = "roberta-large";
local epochs = 10;
local batch_size = 5;

// This defines the number of instances, not the number of questions. One question can end up as
// multiple instances.
local number_of_train_instances = 10685;

{
    "dataset_reader": {
        "type": "squad_eqqa",
        "transformer_model_name": transformer_model
    },
    "train_data_path": "/home/pradeepd/workspace/eqqa/data/squad_v2_eqqa_train.jsonl",
    "validation_data_path": "/home/pradeepd/workspace/eqqa/data/squad_v2_eqqa_dev.jsonl",
    "model": {
        "type": "squad_eqqa",
	"trained_squad_model": "https://storage.googleapis.com/allennlp-public-models/transformer-qa.2021-02-12.tar.gz",
	"two_stage": false
	//"one_class_weight": 0.5,
	//"regression_weight": 0.1,
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
        "cut_frac": 0,
        "num_steps_per_epoch": std.ceil(number_of_train_instances / batch_size)
      },
      "callbacks": [
	{"type": "tensorboard"}
      ],
      "grad_clipping": 1.0,
      "num_epochs": epochs,
      "patience": epochs,
      "validation_metric": "-mae",
      "enable_default_callbacks": false,
      "cuda_device": 0
    }
}
