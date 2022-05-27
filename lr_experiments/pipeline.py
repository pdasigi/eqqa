import numpy as np


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
        #print(f"Loading dataset '{self.dataset}'")
        #if self.dataset == "all":
        #    data = data.copy()
        #else:
        #    data = data[data["dataset"] == self.dataset].copy()

        print(f"Loading dataset '{self.dataset}':", data.dataset.unique())
        self.X_train = data[self.features]
        self.y_train = data[self.target]

    def split(self, holdout_fraction=0.2):
        """"""
        # print(f"Splitting dataset holdout_fraction={holdout_fraction}")
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
            print("Using PCA")
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
        
    def evaluate(self, eval_dataset=None):
        """"""
        import sklearn.metrics as m
        import scipy.stats as st
        
        if eval_dataset is None:
            # print("Evaluating holdout dev set.")
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
        
        print(f"Loading **fewshot** dataset '{self.fewshot_dataset}':", fewshot_data.dataset.unique())
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
        
        weights = np.ones(fewshot_n+ptrain_n)
        weights[:ptrain_n] *= ptrain_w
        weights[ptrain_n:] *= fewshot_w

        X = np.vstack((self.X_train, self.X_fewshot))
        y = np.concatenate((self.y_train, self.y_fewshot))
        self.X_train = X
        
        # Preprocessing 
        self.preprocess() # modifies self.X_train inplace

        self.model = self.model_class(**self.model_hparams)
        self.model.fit(self.X_train, y, sample_weight=weights)
        self.X_train = self.X_train_orig
        
    def evaluate(self, eval_dataset=None):
        results = super().evaluate(eval_dataset=eval_dataset)
        results["fewshot_weight"] = self.fewshot_weight
        results["fewshot_n"] = len(self.y_fewshot)

        return results
