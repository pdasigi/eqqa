rm -rf outputs/models/$1
allennlp train training_config/mocha_eqqa.jsonnet -s outputs/models/$1 --include-package quality_estimation
