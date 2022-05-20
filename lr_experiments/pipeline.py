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
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_train, self.y_train,
            test_size=holdout_fraction, 
            random_state=self.seed, 
            stratify=self.y_train,
        )
            
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test        
        
    def preprocess(self, with_std=True, with_pca=False, **kwargs):
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
            operations.append(PCA(random_state=self.seed, **kwargs))
        
        self.preproc_fn = make_pipeline(*operations) \
            if len(operations) > 0 else NoPreprocessing()
        
        self.preproc_fn.fit(self.X_train)
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
    pass
        
