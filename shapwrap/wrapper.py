from typing import Any, Callable, Tuple, List, Dict
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt



def aggregate_shap_by_group(shap_vals, features:List[str], groups:Dict[str, str]):
    """
    Reference: https://www.kaggle.com/code/estevaouyra/shap-advanced-uses-grouping-and-correlation/notebook
    :param shap_vals:
    :param features:
    :param groups: {'group_key': [list of feature names]}
    :return:
    """
    from itertools import repeat, chain

    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name="features")).T
    shap_Tdf["group"] = shap_Tdf.reset_index().features.map(groupmap).values
    # find a shap-sum (average directionality) per row per group
    shap_grouped = shap_Tdf.groupby("group").sum().T
    return shap_grouped


def column_to_prefix_map(columns: List[str], length_difference_thresh: int=5, delimeter:str="_"):
    """
    Go through list of columns and find groups that start with the same prefix.
    Assume "_" is used as a word delimeter in column names.
    Note, there is a filter to make sure all column groups are roughly the same length. Otherwise, I assume
    it's another feature. (e.g. "fruits_apple", "fruits_orange", "fruits_basket_average_price")
    :param columns:
    :param length_difference_thresh: acceptable deviation in column name length from group average
    :return:
    """
    # feature -> prefix_group mapping:
    prefixes = {}
    for col in columns:
        # remove the last suffix (assuming it's from get_dummies)
        col_clean = delimeter.join(col.split(delimeter)[:-1])
        if col_clean != '':
            prefixes[col_clean] = prefixes.get(col_clean, 0) + 1
        else:
            prefixes[col] = prefixes.get(col, 0) + 1

    prefix_groups = {}
    for prefix, count in prefixes.items():
        affiliated_features = [col for col in columns if prefix in col]
        if count > 1:
             # remove features that look anomalous (longer/shorter than all others)
            # e.g. country_of_origin_gdp_per_capita vs country_of_origin_AD
            average_length = pd.Series(affiliated_features).apply(len).mean()
            affiliated_features = [
                col for col in affiliated_features if np.abs(len(col) - average_length) <= length_difference_thresh
            ]
            if len(affiliated_features) > 1:
                prefix_groups[prefix] = affiliated_features
        else:
            prefix_groups[prefix] = affiliated_features
    return prefix_groups


class ShapExplanation:
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any = None,
        n_sample: int = 10000,
        output_probability: bool = False,
        shap_values_object=None,
    ):
        """
        This class handles the SHAP library API calls.
        :param data:
        :param y: label column-name
        :param model: trained object
        :param n_sample: number of data samples SHAP will work with (to save time on computation)
        :param output_probability: if possible, you can try to output probability values intstead of raw SHAP values to
                                make this more interpretable. But it only works for binary problems for now.
        """
        X = data.copy()
        self.X = X
        self.columns = data.columns
        if shap_values_object is None:
            # remove 'y' from X columns
            # prepare dataset
            # subsample to speed things up
            if len(data) >= n_sample:
                print("Subsampling data to ", n_sample)
                self.X = X.sample(n_sample, random_state=44)
            self.output_probability = output_probability
            self.model = model

            # Shap explanation
            self.shap_values = None
            self.shap_values_object = None
            self.explain_prediction()
        else:
            self.shap_values_object = shap_values_object
            self.shap_values = shap_values_object["values"]

    def get_shap_values(self):
        return self.shap_values_object

    def explain_prediction(self, output="shap"):
        """
        :param output: "regular" is a standard shap output.
                       "probability" is a shap value converted to class probability. This function is less stable and
                       only works for binary problems it seems.
        :return:
        """
        print("Explaining predictions via SHAP:")
        kwargs = {}
        if self.output_probability:
            print("Trying probabilistic SHAP conversion.")
            kwargs["data"] = self.X.astype("float64")
            kwargs["feature_perturbation"] = "interventional"
            kwargs["model_output"] = "probability"

        # explainer by model type
        model_type = type(self.model).__name__
        print("Model is: ", model_type)
        try:
            # Following API documentation at: https://shap.readthedocs.io/en/latest/index.html
            if model_type in ["XGBClassifier", "XGBRegressor"]:  # TODO: add CatBoost, LGBM, RandomForest should be skipped.
                print("Using TreeExplainer")
                explainer = shap.TreeExplainer(self.model, **kwargs)
            elif model_type in ["LinearRegression", "LogisticRegression"]:
                print("Using LinearExplainer")
                kwargs["masker"] = shap.maskers.Impute(data=self.X)
                explainer = shap.LinearExplainer(self.model, **kwargs)
            elif model_type in ["Sequential"]:
                # Example at: https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.html
                # select a set of background examples to take an expectation over
                background = self.X[np.random.choice(self.X.shape[0], 100, replace=False)]
                explainer = shap.DeepExplainer(self.model, background)
            else:
                print(
                    "Using base-class Explainer. WARNING: Probability does not work for this. Also this explainer is a bit unstable, use with caution."
                )
                f = lambda x: self.model.predict_proba(x)
                # referenced their examples, not 100% sure about what med does
                med = self.X.median().values.reshape((1, self.X.shape[1]))
                explainer = shap.Explainer(f, med, **kwargs)
        except Exception as e:
            print(e)
            print("if error is `Model does not have a known objective or output type!`, remove `output_probability` flag.")

        self.shap_values_object = explainer(self.X)
        self.shap_values = self.shap_values_object.values
        return self.shap_values

    def test(self):
        return "success"


    def plot(
        self,
        plot_type: str = "summary",
        n_features_display: int = 50,
        features_ignore=None,
        keep_zero_values=False,
        save_plot_path=None
    ):
        """
        plot_type: "summary", "summary_group", "dependence", "decision_paths", "feature_breakout"
            "summary": (Horizontal axis) is SHAP-value, telling you 1) how much 2) which direction did a particular feature push the decision of the model. (Color) tells you the values of the feature. If feature value (color) is proportional to SHAP-value, it means this feature is independent of other features. 
            "summary_group": "Same as `summary` but will group categorical features and their effects together."
            "dependence": If a feature is dependent on other features, we can plot that relationship. The most dominant interaction is chosen by default (color) to plot feature values (horizontal) and shap values (vertical) against.
            "decision_paths": The way to read the plot is **bottom-up**. Every prediction starts at 'base_value' (expected value of the model) and quickly diverges because certain features push it left or right. Most important features are at the top.
            "feature_breakout": You can see how different feature value contribute to model decision on average.
        features_ignore (List[str], optional): Ignore these features.
        keep_zero_values (bool, optional): If True, features value of zero will be kept in the plot. Relevant for categorical flags, which are usually zero effect if they're not active. Default is False.
        save_plot_path (str, optional): If provided, the plot will be saved at the specified path. Default is None.
        """
        print("PLOTTING `{}`:".format(plot_type))

        shap_values = self.shap_values
        X_plot = self.X
        columns = self.X.columns
        # to find column id from column name
        col_to_id = {col: i for i, col in enumerate(columns)}

        features = self.X.columns
        # filter some features
        if features_ignore is not None:
            features = [x for x in features if x not in features_ignore]
        feature_ids_to_plot = [col_to_id[feat] for feat in features]

        if plot_type == "summary":
            # we provide original data for 1) names of columns 2) data values, compared with 3) Shap values.
            # import pdb;pdb.set_trace()
            plot = shap.summary_plot(
                shap_values[:, feature_ids_to_plot],
                X_plot.iloc[:, feature_ids_to_plot],
                max_display=n_features_display,
                alpha=0.3,
                show=False
            )
            if save_plot_path is not None:
                plt.savefig(save_plot_path)
            plt.show()
            vals = np.abs(shap_values[:, feature_ids_to_plot]).mean(0)
            feature_importance = pd.DataFrame(list(zip(features, vals)), columns=["col_name", "feature_importance_vals"])
            feature_importance.sort_values(by=["feature_importance_vals"], ascending=False, inplace=True)
        elif plot_type == "summary_group":
            prefix_groups = column_to_prefix_map(columns=self.columns)
            shap_groups = aggregate_shap_by_group(shap_vals=self.shap_values, features=self.columns, groups=prefix_groups)
            plot = shap.summary_plot(shap_groups.values, features=shap_groups.columns,  alpha=0.3,
                                     show=False)
            if save_plot_path is not None:
                plt.savefig(save_plot_path)
            plt.show()
            vals = np.abs(shap_values[:, feature_ids_to_plot]).mean(0)
            feature_importance = pd.DataFrame(list(zip(features, vals)), columns=["col_name", "feature_importance_vals"])
            feature_importance.sort_values(by=["feature_importance_vals"], ascending=False, inplace=True)
        elif plot_type == "dependence":
            for feat in features:
                shap.dependence_plot(feat, shap_values, X_plot, alpha=0.3)  # , interaction_index=None
                plt.show()
        elif plot_type == "decision_paths":
            shap.decision_plot(
                self.shap_values_object.base_values[0],
                shap_values[:, feature_ids_to_plot],
                X_plot.iloc[:, feature_ids_to_plot],
                alpha=0.1,
                ignore_warnings=True,
            )
        elif plot_type == "feature_breakout":
            col_types = zip(X_plot.dtypes, X_plot.columns)
            counts = []
            for col in col_types:
                col_type, col_name = col
                if col_name in features:
                    id_in_shap_matrix = col_to_id[col_name]
                    shap_values_for_col = shap_values[:, [id_in_shap_matrix]].flatten()
            
                    # Determine whether a feature is continuous by checking its dtype.
                    is_continuous = np.issubdtype(col_type, np.number)  # Update this condition based on your data.
            
                    feature_values = X_plot[col_name]
                    if is_continuous:
                        # Bin continuous feature values into 3 categories.
                        feature_values = pd.cut(feature_values, bins=3, labels=False)
            
                    df_val_con = pd.DataFrame(
                        zip(feature_values, shap_values_for_col, shap_values_for_col), columns=["value", "size", "mean_impact"]
                    )
                    df_val_con["feature"] = col_name
                    count = (
                        df_val_con.groupby("value")
                        .agg({"feature": "first", "size": "count", "mean_impact": "mean"})
                        .reset_index(0)
                    )
                    count["size"] = count["size"] / len(X_plot)
                    count["mean_impact*size"] = count["mean_impact"] * count["size"]
                    counts.append(count)
            
            counts = pd.concat(counts)
            columns_order = ["feature", "value", "mean_impact*size", "mean_impact", "size"]
            counts = counts[columns_order]
            
            # if not keep_zero_values:
            #     counts = counts.loc[counts["value"] != 0, :]
            
            counts_pretty = counts.reset_index().style.background_gradient(subset=["mean_impact*size", "mean_impact", "size"])
            display(counts_pretty)
        else:
            print("Wrong plot type")