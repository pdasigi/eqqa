import numpy as np
import pandas as pd
import logging 


class NoPreprocessing:
    def fit(self, *args, **kwargs): 
        pass
    
    def transform(self, X, *args, **kwargs): 
        return X
    
    def fit_transform(self, X, *args, **kwargs): 
        return X
    

class Pipeline:
    """"""
    def __init__(self, model_class, model_hparams, dataset, features, target, seed=81263):
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.model = self.model_class(**self.model_hparams)

        self.dataset = dataset
        self.features = features
        self.target = target
        self.seed = seed
        
    def load_data(self, data):
        """"""
        logging.info(f"Loading dataset '{self.dataset}':", data.dataset.unique())
        self.X_train = data[self.features]
        self.y_train = data[self.target]

    def split(self, holdout_fraction=0.2):
        """"""
        logging.debug(f"Splitting dataset holdout_fraction={holdout_fraction}")
        from sklearn.model_selection import train_test_split
        
        if holdout_fraction == 0 or holdout_fraction == 1:
            self.X_train, self.X_test = self.X_train, None
            self.y_train, self.y_test = self.y_train, None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train,
            test_size=holdout_fraction, 
            random_state=self.seed, 
            stratify=self.y_train,
        )
            
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test        
        
    def preprocess(self, with_std=True, with_pca=False, pca_kwargs={}, **kwargs):
        """"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.decomposition import PCA
        
        operations = []
        
        if with_std:
            # print("Using StandardScaler")
            operations.append(StandardScaler())
        if with_pca:
            logging.debug("Using PCA")
            operations.append(PCA(random_state=self.seed, **pca_kwargs))
        
        self.preproc_fn = make_pipeline(*operations) \
            if len(operations) > 0 else NoPreprocessing()
        
        self.preproc_fn.fit(self.X_train, **kwargs)
        self.X_train = self.preproc_fn.transform(self.X_train)
        
        if getattr(self, "X_test", None) is not None:
            self.X_test = self.preproc_fn.transform(self.X_test)
        
    def fit(self):
        """"""
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        X = self.preproc_fn.transform(X[self.features])
        return self.model.predict(X)

    def evaluate(self, eval_dataset=None):
        """"""
        import sklearn.metrics as m
        import scipy.stats as st
        
        if eval_dataset is None:
            logging.info("Evaluating holdout dev set.")
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test = eval_dataset[self.features]
            y_test = eval_dataset[self.target]    
            X_test = self.preproc_fn.transform(X_test)
        
        # Evaluation
        scores = self.model.predict(X_test)
        
        return {
            "mse": m.mean_squared_error(y_pred=scores, y_true=y_test),
            "r2": m.r2_score(y_pred=scores, y_true=y_test),
            "pearson": st.pearsonr(scores, y_test)[0],
            "spearman": st.spearmanr(scores, y_test)[0],
            "kendalltau": st.kendalltau(scores, y_test)[0],
            "features": self.features,
            "target": self.target,
            "model_classpath": str(self.model_class.__name__),
            "model_hparams": str(self.model_hparams),
            "trained_on": self.dataset,
        }
        
    def evaluate_multiple(self, eval_datasets: dict):
        all_results = []
        
        for name, eval_dataset in eval_datasets.items():
            eval_result = self.evaluate(eval_dataset)
            eval_result["evaluated_on"] = name
            all_results.append(eval_result)
            
        return all_results


class FewShotPipeline(Pipeline):
    def __init__(self, fewshot_dataset, fewshot_weight, **kwargs):
        super().__init__(**kwargs)
        self.fewshot_dataset = fewshot_dataset
        # pre-training to fewshot weight
        # a value of 0.1 implies that the fewshot dataset
        # worths 10% of the dataset and the pretraining set is worth 90%
        assert fewshot_weight is None or 0 < fewshot_weight <= 1, "fewshot_weight should be in (0, 1)"
        self.fewshot_weight = fewshot_weight
        
    def load_data(self, data, fewshot_data):
        super().load_data(data)
        self.X_train_orig = self.X_train[self.features].copy()
        
        logging.info(f"Loading **fewshot** dataset '{self.fewshot_dataset}':", fewshot_data.dataset.unique())
        self.X_fewshot = fewshot_data[self.features].copy()
        self.y_fewshot = fewshot_data[self.target]

    def _compute_weights(self, n_fewshot, n_ptrain):
        if self.fewshot_weight is None:
            return 1, 1
        elif n_fewshot == 0:
            return 0, 1 - self.fewshot_weight
        elif n_ptrain == 0:
            return self.fewshot_weight, 0

        n = n_ptrain + n_fewshot
        target_fewshot = self.fewshot_weight * n
        target_ptrain  = (1-self.fewshot_weight) * n
        
        fewshot_weight = target_fewshot / n_fewshot
        ptrain_weight  = target_ptrain / n_ptrain
        
        logging.info(f"FS_weight={fewshot_weight}, PT_weight={ptrain_weight}")
        return fewshot_weight, ptrain_weight
        
    def _prepare_data(self):
        X, y = [], []
        if len(self.X_train) != 0:
            X.append(self.X_train)
            y.append(self.y_train)
        if len(self.X_fewshot) != 0:
            X.append(self.X_fewshot)
            y.append(self.y_fewshot)
        
        X = np.vstack(X)
        y = np.concatenate(y)
        return X, y

    def fewshot_fit(self):       
        fewshot_n, ptrain_n = len(self.X_fewshot), len(self.X_train)
        fewshot_w, ptrain_w = self._compute_weights(fewshot_n, ptrain_n)
        
        weights = np.ones(fewshot_n+ptrain_n)
        weights[:ptrain_n] *= ptrain_w
        weights[ptrain_n:] *= fewshot_w

        X, y = self._prepare_data()
        self.X_train = X
        
        # Preprocessing
        self.X_train = pd.DataFrame(self.X_train, columns=self.features)
        self.preprocess() # modifies self.X_train inplace

        self.model = self.model_class(**self.model_hparams)
        self.model.fit(self.X_train, y, sample_weight=weights)
        self.X_train = self.X_train_orig
        
    def evaluate(self, eval_dataset=None):
        results = super().evaluate(eval_dataset=eval_dataset)
        results["fewshot_weight"] = self.fewshot_weight
        results["fewshot_n"] = len(self.y_fewshot)

        return results


class FineTuningFewShotPipeline(Pipeline):
    def __init__(self, fewshot_dataset, fewshot_weight, **kwargs):
        super().__init__(**kwargs)
        self.fewshot_dataset = fewshot_dataset
        # pre-training to fewshot weight
        # a value of 0.1 implies that the fewshot dataset
        # worths 10% of the dataset and the pretraining set is worth 90%
        assert fewshot_weight is None or 0 < fewshot_weight <= 1, "fewshot_weight should be in (0, 1)"
        self.fewshot_weight = fewshot_weight
        
    def load_data(self, data, fewshot_data):
        super().load_data(data)
        self.X_train_orig = self.X_train[self.features].copy()
        
        logging.info(f"Loading **fewshot** dataset '{self.fewshot_dataset}':", fewshot_data.dataset.unique())
        self.X_fewshot = fewshot_data[self.features].copy()
        self.y_fewshot = fewshot_data[self.target]
        
    def _compute_weights(self, n_fewshot, n_ptrain):
        if self.fewshot_weight is None:
            return 1, 1

        n = n_ptrain + n_fewshot
        target_fewshot = self.fewshot_weight * n
        target_ptrain  = (1-self.fewshot_weight) * n
        
        fewshot_weight = target_fewshot / n_fewshot
        ptrain_weight  = target_ptrain / n_ptrain
        
        return fewshot_weight, ptrain_weight
        
    def fewshot_fit(self):       
        fewshot_n, ptrain_n = len(self.X_fewshot), len(self.X_train)
        fewshot_w, ptrain_w = self._compute_weights(fewshot_n, ptrain_n)
        
        # Fit base classifier
        rand = np.random.default_rng(self.seed)
        X_pretrain_ix = np.arange(self.X_train.shape[0])
        X_pretrain_ix = rand.choice(X_pretrain_ix, size=ptrain_n, replace=False)
        self.X_train, self.y_train = self.X_train.values[X_pretrain_ix], self.y_train.values[X_pretrain_ix]
        
        # Preprocessing 
        self.preprocess() # modifies self.X_train inplace
        self.model = self.model_class(**self.model_hparams)
        self.model.fit(self.X_train, self.y_train)
        self.X_train = self.X_train_orig

        # Fine tuning
        X_fewshot_ix = np.arange(len(self.y_fewshot))
        X_fewshot_ix = rand.choice(X_fewshot_ix, size=fewshot_n, replace=False)
        X_fewshot, y_fewshot = self.X_fewshot.values[X_fewshot_ix], self.y_fewshot.values[X_fewshot_ix]
        X_fewshot = self.preproc_fn.transform(X_fewshot)
        self.model.fit(X_fewshot, y_fewshot)
        
    def evaluate(self, eval_dataset=None):
        results = super().evaluate(eval_dataset=eval_dataset)
        results["fewshot_weight"] = self.fewshot_weight
        results["fewshot_n"] = len(self.y_fewshot)

        return results


if __name__ == "__main__":
    # compute zero shot LERC
    from create_datasets import read_json_dataset
    from dict_utils import unfold_to_list, fold_from_list

    import pandas as pd
    import numpy as np 

    def add_lerc_preds(data, lerc_preds_dir, split):
        lerc_preds = read_json_dataset(lerc_preds_dir, split)
            
        for dataset, d in lerc_preds.items():
            for example_id, score in d.items():
                data[dataset][example_id]["LERC"] = (score["pred_score"] - 1) / (5-1)
                
        return data

    DATASET_DIR = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/lr_experiments"

    RESULTS_DIR = "/home/kat/Projects/PhD/qasper-experiments/eqqa/lr_experiments/experiments_20220607/results/zero-shot"
    IMAGES_DIR = "/home/kat/Projects/PhD/qasper-experiments/eqqa/lr_experiments/experiments_20220607/images/zero-shot"

    TRAIN_LOO_DATASETS = {}
    FEW_SHOT_LOO_DATASETS = {}
    DEV_LOO_DATASETS = {}

    for dataset in ["narrativeqa", "mcscript"]:
        train = read_json_dataset(DATASET_DIR, "train_metrics")
        dev = read_json_dataset(DATASET_DIR, "dev_metrics")
        print(len(train), len(dev))
        
        LERC_PREDS_DIR = f"{DATASET_DIR}/lerc_wo_{dataset}"
        add_lerc_preds(train, LERC_PREDS_DIR, "train_preds")
        add_lerc_preds(dev, LERC_PREDS_DIR, "dev_preds")
    
        train_df = pd.DataFrame(unfold_to_list(train, "dataset", "example_id"))
        dev_df   = pd.DataFrame(unfold_to_list(dev, "dataset", "example_id"))
        
        # Scale the scores (we will have negatives and above 1)
        train_df["score_scaled"] = train_df.score.apply(lambda s: (s-1)/(5-1))
        dev_df["score_scaled"] = dev_df.score.apply(lambda s: (s-1)/(5-1))
        
        
        # Compute the training leave-one-out (all datasets except the LOO)
        train_loo_df = train_df[train_df["dataset"] != dataset]  
        TRAIN_LOO_DATASETS[f"except_{dataset}"] = train_loo_df
        
        # Compute the few shot dataset (the LOO training split)
        fewshot_loo_df = train_df[train_df["dataset"] == dataset]
        FEW_SHOT_LOO_DATASETS[f"{dataset}"] = fewshot_loo_df

        # Development set, we'll evaluate on the narrativeQA directly
        dev_df   = dev_df[dev_df["dataset"] == dataset]
        DEV_LOO_DATASETS[dataset] = dev_df
        
        print(train_loo_df.shape, fewshot_loo_df.shape, dev_df.shape)
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from lightgbm import LGBMRegressor


    def run_few_shot_experiment(
        train_datasets,
        dataset_name,
        fewshot_datasets,
        fewshot_dataset_name,
        eval_datasets,
        features,
        target,
        nruns=5,
        seed=81723,
        model_class=LinearRegression,
        model_hparams={},
        pipeline=None,
        pretrain_pct = [1],
        fewshot_n_examples = None,
        fewshot_weights=[None],
    ):
        from itertools import product
        rand = np.random.default_rng(seed)
        seed_fn = lambda rand: int(rand.integers(10**6))
        
        pretrain_data = train_datasets[dataset_name]
        pretrain_n_examples = [len(pretrain_data) * n for n in pretrain_pct]
        pretrain_n_examples = [int(round(n, 0)) for n in pretrain_n_examples]

        fewshot_data = fewshot_datasets[fewshot_dataset_name]
        print(len(fewshot_data))
        fewshot_n_examples = [n for n in fewshot_n_examples if n <= len(fewshot_data)]
        
        print("#PT:", pretrain_n_examples)
        print("#FS:", fewshot_n_examples)
        print("FS Weights:", fewshot_weights)

        all_results = []
        all_pipelines = []
        print(list(product(pretrain_n_examples, fewshot_n_examples, fewshot_weights)))
        for i, (pretrain_n, fewshot_n, fewshot_w) in enumerate(product(pretrain_n_examples, fewshot_n_examples, fewshot_weights)):
            for j in range(nruns):
                pretrain_fraction = pretrain_data.sample(n=pretrain_n, replace=False, random_state=seed_fn(rand))
                
                seed = rand.integers(10**6)
                fewshot_fraction =  fewshot_data.sample(n=fewshot_n, replace=False, random_state=seed_fn(rand))

                # Get subset of few shot data:
                if pipeline is None:
                    pipeline = FewShotPipeline

                fs_pipeline = pipeline(
                    fewshot_dataset=fewshot_dataset_name,
                    fewshot_weight=fewshot_w,
                    model_class=model_class,
                    model_hparams=model_hparams,
                    dataset=dataset_name,
                    features=features,
                    target=target,
                    seed=seed_fn(rand),
                )

                fs_pipeline.load_data(pretrain_fraction, fewshot_data=fewshot_fraction)
                fs_pipeline.fewshot_fit()
                results = fs_pipeline.evaluate_multiple(eval_datasets)
                
                for r in results:
                    r["i"] = i
                    r["seed"] = seed_fn(rand)
                    r["pretrain_n"] = pretrain_n
                    r["pretrain_pct"] = round(pretrain_n / len(pretrain_data), 2)
                    r["fewshot_n"] = fewshot_n
                    r["fewshot_weight"] = fewshot_w
                    
                all_results.extend(results)
                all_pipelines.append(fs_pipeline)
                
        return all_results, all_pipelines

    class Model: 
        def __init__(self):
            self.name = None
            self.classpath = None
            self.pipeline = None
            self.nruns = None
            
        def __str__(self):
            return f"{self.name}_{self.nruns}"
    

    model = Model()
    model.name, model.classpath, model.hparams, model.nruns = "lr", LinearRegression, {}, 10
    #model.name, model.classpath, model.hparams, model.nruns = "lgbm", LGBMRegressor, {"random_state": 113}, 10
    # model.name, model.classpath, model.hparams, model.nruns = "rf", RandomForestRegressor, {"n_jobs": 15}, 1
    # model.name, model.classpath, model.hparams, model.nruns, model.pipeline = "mlp", MLPRegressor, {"learning_rate": "adaptive", "random_state": 42, "early_stopping": True}, 5, FineTuningFewShotPipeline


    METRICS = [
        # Bleu
        'bleu1', 'bleu2', 'bleu3', 'bleu4', 
        # 'hf_bleu1', 'hf_bleu2', 'hf_bleu3', 'hf_bleu4', 
        'rougeL', 
        # 'hf_rougeL', 'hf_rougeLsum',
        'hf_rouge1', 'hf_rouge2',
        'meteor',
        'recall', 'precision', 'f1_score',
        'sari_context', 'sari_question',
        # Token overlap when 1st error occurred
        'precision_at_err1', 'recall_at_err1',
        # Confusion matrix
        'tp', 'fn', 'fp',
        # Edit scores ------
        'char_edit_score', 'word_edit_score',
        # Learned metrics -------
        'bertscore', 
        'bleurt',
        "LERC",
        # Input statistics ------
        'candidatelength_word', 'candidatelength_char',
        'candidatenunique_words', 'referencelength_word',
        'referencelength_char', 'referencenunique_words',
        'contextlength_word', 'contextlength_char',
        'contextnunique_words', 'questionlength_word',
        'questionlength_char', 'questionnunique_words',
    ]

    if "bleurt" not in METRICS:
        model.name += '_no_bleurt'
    if "bertscore" not in METRICS:
        model.name += "_no_bertscore"
    if "LERC" in METRICS:
        model.name += "_with_LERC"
        
    print(model)

    ALPHAS = [None, 0.0]
    FEWSHOT_WEIGHTS = [round(1-alpha, 2) if isinstance(alpha, (int, float)) else alpha for alpha in ALPHAS]
    PRETRAIN_PCTS = [1]
    FEWSHOT_N = [1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 48, 64, 96, 128, 200, 256, 512, 1024]

    print(FEWSHOT_WEIGHTS, PRETRAIN_PCTS, FEWSHOT_N)

    for dataset in ["narrativeqa",]:
        # for dataset in DATASETS:
            print("\n" * 8, "Experiment for dataset", dataset)
            loo_fewshot, loo_ps =  run_few_shot_experiment(
                train_datasets=TRAIN_LOO_DATASETS,
                dataset_name=f"except_{dataset}",
                fewshot_datasets=FEW_SHOT_LOO_DATASETS,
                fewshot_dataset_name=dataset,
                eval_datasets=DEV_LOO_DATASETS,
                features=METRICS,
                target="score_scaled",
                nruns=1,
                seed=81723,
                model_class=model.classpath,
                model_hparams=model.hparams,
                pipeline=model.pipeline,
                # ----------------------------------------
                # few shot parameters
                # ----------------------------------------
                pretrain_pct=PRETRAIN_PCTS,
                fewshot_n_examples=FEWSHOT_N,
                fewshot_weights=FEWSHOT_WEIGHTS,
            )
            loo_results = pd.DataFrame(loo_fewshot)
            loo_results.fewshot_weight = loo_results.fewshot_weight.fillna("default")
