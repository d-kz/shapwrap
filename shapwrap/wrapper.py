from typing import Any, Callable, Tuple, List, Dict
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt

cm = sns.light_palette("blue", as_cmap=True)
cm_red = sns.light_palette("red", as_cmap=True)
# cm = sns.diverging_palette(5, 250, as_cmap=True)


def shap_importance_by_cluster(data, shap_values, cluster_assignment):
    # plot 2 matrices:
    # 1) feature average to give a sense of what type of cluster it is
    # 2) shap average to give a sense of which way features are contributing

    df_clusters = data.copy()
    df_shap = pd.DataFrame(shap_values, columns=data.columns)
    # assign clusters to rows
    df_shap["cluster"] = cluster_assignment
    df_shap["shap_sum"] = shap_values.sum(axis=1)  # sum of explanation vector
    df_clusters["cluster"] = cluster_assignment
    df_clusters["shap_sum"] = shap_values.sum(axis=1)  # sum of explanation vector

    # Summarize the cluster by the average feature "portrait"
    # mean for shap
    # count for cluster
    col_types = zip(data.columns, data.dtypes)

    aggs = {}
    for feature, col_type in col_types:
        # TODO: maybe add categoric columns as param
        # mode for integer 'class-encoding' type features (we want the most common class, mean makes no sense since the feature is not continuous)
        # if str(col_type) in ['int8', 'int64']:
        #     aggs[feature] = lambda x: scipy.stats.mode(x)[0]
        # # mean for continuous/boolean (average makes sense here)
        # else:
        aggs[feature] = "mean"

    # these stay the same always
    aggs["cluster"] = lambda x: float(len(x)) / len(data)
    aggs["shap_sum"] = "mean"
    df_clusters_agg = df_clusters.groupby("cluster").aggregate(aggs)

    df_clusters_agg["impact"] = df_clusters_agg["cluster"] * df_clusters_agg["shap_sum"]
    df_clusters_agg["impact"] = df_clusters_agg["cluster"] * df_clusters_agg["shap_sum"]
    df_clusters_agg = df_clusters_agg.rename(columns={"shap_sum": "mean_impact", "cluster": "size"})
    df_clusters_agg = df_clusters_agg.sort_values("mean_impact").astype(float).round(2)

    # print average shap vector for each cluster
    df_shap = df_shap.groupby("cluster").aggregate("mean")
    # df_shap['impact'] = df_shap['cluster']*df_shap['shap_sum']
    df_shap = df_shap.rename(columns={"shap_sum": "mean_impact"})
    df_shap = df_shap.sort_values("mean_impact").astype(float).round(3)
    df_shap["size"] = df_clusters_agg["size"]
    df_shap["impact"] = df_clusters_agg["impact"]

    # sort columns in order of importance and relevance
    # sum of absolute values of shap values as measure of importance
    ordered_columns = list(zip(np.abs(shap_values).sum(axis=0), data.columns))
    ordered_columns = sorted(ordered_columns, key=lambda x: -x[0])
    ordered_columns = [value_tuple[1] for value_tuple in ordered_columns]
    # ordered_columns = ['impact', 'mean_impact', 'size'] + ordered_columns

    df_clusters_agg = df_clusters_agg[["impact", "mean_impact", "size"] + ordered_columns]
    df_shap = df_shap[["impact", "mean_impact", "size"] + ordered_columns]

    cmap = cm.get_cmap("coolwarm", 10000)

    def ref_color(s, df_shap):
        """
        df_shap is used as shap values reference. We fill in color intensity for average_feature_matrix based on shap values for each cell.
        :param s:
        :param df_shap:
        :return:
        """

        field = s.name
        shap_vals = df_shap[field]
        if field in ["impact", "mean_impact", "size"]:
            return ["" for i in shap_vals]
        # return ['background-color:{};'.format(matplotlib.colors.rgb2hex(cmap(i*100))) for i in shap_vals]
        return ["background-color:red; opacity: {}%;".format(i * 100) for i in shap_vals]

    df_clusters_agg = df_clusters_agg.style.apply(ref_color, df_shap=df_shap, axis=0)

    # df_clusters_agg = (df_clusters_agg.reset_index().style
    #  .background_gradient(cmap="coolwarm")
    #  .background_gradient(subset=['impact', 'mean_impact', 'size', 'cluster'], cmap="PuBu")
    #  )
    df_shap = df_shap.reset_index().style.background_gradient(cmap="PuBu")

    return df_clusters_agg, df_shap


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
        prefixes[col_clean] = prefixes.get(col_clean, 0) + 1

    prefix_groups = {}
    for prefix, count in prefixes.items():
        affiliated_features = [col for col in columns if prefix in col]
        # remove features that look anomalous (longer/shorter than all others)
        # e.g. country_of_origin_gdp_per_capita vs country_of_origin_AD
        average_length = pd.Series(affiliated_features).apply(len).mean()
        affiliated_features = [
            col for col in affiliated_features if np.abs(len(col) - average_length) <= length_difference_thresh
        ]
        if len(affiliated_features) > 1:
            prefix_groups[prefix] = affiliated_features
        else:
            # if only a single column with such name, just use full name
            prefix_groups[affiliated_features[0]] = affiliated_features
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
        features: List[str] = None,
        filter_mask=None,
        n_features_display: int = 50,
        plot_guide: bool = False,
        min_cluster_size: int = 50,
        features_ignore=None,
        keep_zero_values=False,
        save_plot_path=None
    ):
        # TODO: add alpha to all plots
        print("PLOTTING `{}`:".format(plot_type))
        plot_guides = {
            "summary": """(Horizontal axis) is SHAP-value, telling you 1) how much 2) which direction did a particular feature push the decision of the model. (Color) tells you the values of the feature. If feature value (color) is proportional to SHAP-value, it means this feature is independent of other features. """,
            "dependence": """If a feature is dependent on other features, we can plot that relationship. The most dominant interaction is chosen by default (color) to plot feature values (horizontal) and shap values (vertical) against.""",
            "decision_paths": """The way to read the plot is **bottom-up**. Every prediction starts at 'base_value' (expected value of the model) and quickly diverges because certain features push it left or right. Most important features are at the top.""",
            "feature_breakout": """You can see how different feature value contribute to model decision on average.""",
            "shap_clusters": """Since shap-explanation-vectors are a way the model sees the data, we cluster them to 1) show common patterns of the data and 2) sort them in order of impact (mean_impact * size). NOTE: you can choose which features to clusters by.""",
        }
        if plot_guide:
            print("[PLOT EXPLANATION]: ", plot_guides[plot_type])

        shap_values = self.shap_values
        X_plot = self.X
        if filter_mask is not None:
            size_init = len(X_plot)
            subset_ids = X_plot.reset_index().index[filter_mask]
            X_plot = X_plot.iloc[subset_ids]
            shap_values = shap_values[subset_ids]
            print("Filtered set is {} of original:".format(len(X_plot) / size_init))

        # TODO:
        # alpha = 0.0001/len(self.X) # to get roughly same level of transparency for readability.

        columns = self.X.columns
        # to find column id from column name
        col_to_id = {col: i for i, col in enumerate(columns)}

        # plot all features if none specified
        if features == None:
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
            vals = np.abs(shap_values[:, feature_ids_to_plot]).mean(0)
            feature_importance = pd.DataFrame(list(zip(features, vals)), columns=["col_name", "feature_importance_vals"])
            feature_importance.sort_values(by=["feature_importance_vals"], ascending=False, inplace=True)
            return feature_importance
        elif plot_type == "summary_group":
            prefix_groups = column_to_prefix_map(columns=self.columns)
            shap_groups = aggregate_shap_by_group(shap_vals=self.shap_values, features=self.columns, groups=prefix_groups)
            plot = shap.summary_plot(shap_groups.values, features=shap_groups.columns,  alpha=0.3,
                                     show=False)
            if save_plot_path is not None:
                plt.savefig(save_plot_path)

            vals = np.abs(shap_values[:, feature_ids_to_plot]).mean(0)
            feature_importance = pd.DataFrame(list(zip(features, vals)), columns=["col_name", "feature_importance_vals"])
            feature_importance.sort_values(by=["feature_importance_vals"], ascending=False, inplace=True)
            return feature_importance
        elif plot_type == "dependence":
            assert features != None
            for feat in features:
                shap.dependence_plot(feat, shap_values, X_plot, alpha=0.3)  # , interaction_index=None
                # TODO; add # shap.approximate_interactions("Age", shap_values, X)
            return None
        elif plot_type == "decision_paths":
            shap.decision_plot(
                self.shap_values_object.base_values[0],
                shap_values[:, feature_ids_to_plot],
                X_plot.iloc[:, feature_ids_to_plot],
                alpha=0.1,
                ignore_warnings=True,
            )
            return None
        # elif plot_type == 'force_plot':
        #     shap.initjs()
        #     shap.force_plot(shap_values_object.base_values[0],
        #                     shap_values,
        #                     X_plot)
        elif plot_type == "feature_breakout":
            col_types = zip(X_plot.dtypes, X_plot.columns)
            counts = []
            for col in col_types:
                col_type, col_name = col
                # check if user wants to filter for particular features
                if col_name in features:
                    # TODO: bin continuous feature by percentile
                    # we don't want to aggregate continuous features, so only doing this for bool/ints for now.
                    id_in_shap_matrix = col_to_id[col_name]
                    shap_values_for_col = shap_values[:, [id_in_shap_matrix]].flatten()
                    df_val_con = pd.DataFrame(
                        zip(X_plot[col_name], shap_values_for_col, shap_values_for_col), columns=["value", "size", "mean_impact"]
                    )
                    df_val_con["feature"] = col_name
                    count = (
                        df_val_con.groupby("value")
                        .agg({"feature": "first", "size": "count", "mean_impact": "mean"})
                        .reset_index(0)
                    )
                    count["size"] = count["size"] / len(X_plot)
                    count["impact"] = count["mean_impact"] * count["size"]
                    counts.append(count)
            # aggregate results
            counts = pd.concat(counts)
            columns_order = ["feature", "value", "impact", "mean_impact", "size"]
            counts = counts[columns_order]

            # filter out base cases for binary features. TODO: subtract base case effect from positive.
            if not keep_zero_values:
                counts = counts.loc[counts["value"] != 0, :]
            # counts.plot.bar(rot=45, y='contribution_mean')
            counts_pretty = counts.reset_index().style.background_gradient(subset=["impact", "mean_impact", "size"], cmap=cm)
            # display(counts_pretty)
            return counts
        elif plot_type == "feature_split":
            # pre-filter by (feature, feature_value)
            # split by (shap_value)
            # sort columnsin order of statistically significant difference. Or even distribution KL difference, since we can have some bimodal stuff going on.
            size_init = len(X_plot)
            for feat in features:
                feature_id = col_to_id[feat]
                shap_split_val = 0.0
                # right = shap_values[feature_id] >= shap_split_val
                left = shap_values[:, feature_id] < shap_split_val
                X_plot["shap_split"] = "shap_greater_than_{}".format(shap_split_val)
                X_plot.loc[left, "shap_split"] = "shap_less_than_{}".format(shap_split_val)

                average_features = X_plot.groupby("shap_split").aggregate("mean")
                # display(average_features)
                # import pdb;pdb.set_trace()

                def display_sorted(diff, average_features, order=""):
                    print("Ordered by :", order)
                    ordered_columns = zip(diff, average_features.columns)
                    ordered_columns = sorted(ordered_columns, key=lambda x: -x[0])
                    ordered_columns = [value_tuple[1] for value_tuple in ordered_columns]
                    average_features = average_features[ordered_columns]
                    average_features = average_features.reset_index().style.background_gradient(cmap=cm)
                    # display(average_features)

                # # sort by relative absolute difference
                # diff = np.array(average_features)
                # diff = np.abs(diff[0,:] - diff[1,:] /(diff[0,:] + diff[1,:]))
                # display_sorted(diff, average_features, 'relative absolute difference')
                # diff = np.array(average_features)
                # diff = np.abs(diff[0,:] - diff[1,:])
                # display_sorted(diff, average_features, 'absolute difference')
                # TODO: remove into params
                important_features = []
                average_features_select = average_features[important_features]
                diff = np.array(average_features[important_features])
                diff = np.abs(diff[0, :] - diff[1, :])
                display_sorted(diff, average_features_select, "important features, absolute difference")
        # elif plot_type == 'cluster':
        #     from lib.tools.clustering import HighDimensionalNoiseStableClustering
        #
        #     shap_values = shap_values[:, feature_ids_to_plot]
        #     X = X_plot.iloc[:, feature_ids_to_plot]
        #     clusterer = HighDimensionalNoiseStableClustering(data=shap_values,
        #                                                      min_cluster_size=min_cluster_size)
        #     hdb_clusters = clusterer.clusters
        #     df_clusters_agg, df_shap = shap_importance_by_cluster(data=X, shap_values=shap_values, cluster_assignment=hdb_clusters)
        #     display(df_clusters_agg)
        #     display(df_shap)
        #     return df_clusters_agg
        else:
            print("Wrong plot type")


# helloo