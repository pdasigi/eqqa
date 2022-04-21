rm -rf outputs/$1
allennlp train training_config/mocha_eqqa.jsonnet -s outputs/$1 --include-package quality_estimation
