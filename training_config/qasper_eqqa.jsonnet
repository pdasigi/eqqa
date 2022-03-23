local transformer_model = "allenai/led-base-16384";
local epochs = 10;
local batch_size = 1;
local num_gradient_accumulation_steps = 2;

//local train_data_path = "/home/pradeepd/workspace/eqqa/data/qasper_eqqa_data/qasper_eqqa_train.json";
//local train_data_path = "/home/pradeepd/workspace/eqqa/data/qasper_eqqa_data/qasper_eqqa_40p_model_train.json";
local train_data_path = "/home/pradeepd/workspace/eqqa/data/qasper_eqqa_data/qasper_eqqa_dev_split1.json";
local dev_data_path = "/home/pradeepd/workspace/eqqa/data/qasper_eqqa_data/qasper_eqqa_dev_split2.json";

//local training_data_size = 2672;
local training_data_size = 520;
local num_gpus = 1;

local trained_qasper_model = "https://pradeepd-qasper.s3.us-west-2.amazonaws.com/qasper_trained_models/qasper_trained_led_base_hf_serialized.tgz";
//local trained_qasper_model = "/home/pradeepd/data/qasper_models/qasper_led_base_40p_hf_serialized.tgz";

{
    "dataset_reader": {
        "type": "qasper_eqqa",
        "transformer_model_name": transformer_model
    },
    "train_data_path": train_data_path,
    "validation_data_path": dev_data_path,
    "vocabulary": {
        "type": "empty"
    },
    "model": {
        "type": "qasper_eqqa",
	"trained_qasper_model": trained_qasper_model,
	"encoder_output_projector": {
		"input_dim": 768,
	        "num_layers": 2,
		"hidden_dims": [384, 1],
		"activations": "tanh",
		"dropout": 0.1
	},
	"decoder_output_projector": {
		"input_dim": 768,
	        "num_layers": 2,
		"hidden_dims": [384, 1],
		"activations": "tanh",
		"dropout": 0.1
	}
    },
    "data_loader": {
        "batch_size": batch_size
    },
    "trainer": {
      "optimizer": {
        "type": "adam",
        "lr": 5e-5
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
    "pytorch_seed": 15371,
}
