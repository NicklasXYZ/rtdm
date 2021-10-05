import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from stream_handler import setup
setup()
import pickle

import backend.detectorevaluator as evaluator
import backend.geohashsequenceinterpolator as ip
import numpy as np
import pandas as pd


bits_per_char = 2
center = [55.3923666589426, 10.386784608945067]
user = "d1b0c137-1cd0-4430-ad88-4dc24c88f3ad"

def extract_stats(results, method, aggregate=False):
    _data = []
    if method == "dd0":
        if aggregate == False:
            # Collect results across different batches for method dd1
            for params, experiment in results:
                for batch in experiment:
                    dict_ = {
                        "batch": batch,
                        "precision": params["precision"],
                        "theta": params["theta"],
                        "score": experiment[batch]["score"],
                        "median_fit_time": experiment[batch]["median_fit_time"],
                        "median_detect_time": experiment[batch][
                            "median_detect_time"
                        ],
                        "median_delay": experiment["median_delay"],
                    }
                    _data.append(dict_)
        else:
            # Collect results across different batches for method dd1
            for params, experiment in results:
                dict_ = {
                    "precision": params["precision"],
                    "theta": params["theta"],
                    "score": [],
                    "median_fit_time": [],
                    "median_detect_time": [],
                    "median_delay": [],
                }
                for batch in experiment:
                    dict_["score"].append(experiment[batch]["score"])
                    dict_["median_fit_time"].append(
                        experiment[batch]["median_fit_time"]
                    )
                    dict_["median_detect_time"].append(
                        experiment[batch]["median_detect_time"]
                    )
                    dict_["median_delay"].append(
                        experiment[batch]["median_delay"]
                    )
                dict_["score"] = np.median(dict_["score"])
                dict_["median_fit_time"] = np.median(dict_["median_fit_time"])
                dict_["median_detect_time"] = np.median(
                    dict_["median_detect_time"]
                )
                dict_["median_delay"] = np.median(dict_["median_delay"])
                _data.append(dict_)
    elif method == "dd1":
        if aggregate == False:
            # Collect results across different batches for method dd1
            for params, experiment in results:
                for batch in experiment:
                    dict_ = {
                        "batch": batch,
                        "precision": params["precision"],
                        "threshold_high": params["threshold_high"],
                        "score": experiment[batch]["score"],
                        "median_fit_time": experiment[batch]["median_fit_time"],
                        "median_detect_time": experiment[batch][
                            "median_detect_time"
                        ],
                        "median_delay": experiment["median_delay"],
                    }
                    _data.append(dict_)
        else:
            # Collect results across different batches for method dd1
            for params, experiment in results:
                dict_ = {
                    "precision": params["precision"],
                    "threshold_high": params["threshold_high"],
                    "score": [],
                    "median_fit_time": [],
                    "median_detect_time": [],
                    "median_delay": [],
                }
                for batch in experiment:
                    dict_["score"].append(experiment[batch]["score"])
                    dict_["median_fit_time"].append(
                        experiment[batch]["median_fit_time"]
                    )
                    dict_["median_detect_time"].append(
                        experiment[batch]["median_detect_time"]
                    )
                    dict_["median_delay"].append(
                        experiment[batch]["median_delay"]
                    )

                dict_["score"] = np.median(dict_["score"])
                dict_["median_fit_time"] = np.median(dict_["median_fit_time"])
                dict_["median_detect_time"] = np.median(
                    dict_["median_detect_time"]
                )
                dict_["median_delay"] = np.median(dict_["median_delay"])
                _data.append(dict_)
    return pd.DataFrame(data=_data)


def eval_dd0(df, user, params):
    de = evaluator.LeaveOneOutDetectorEvaluator(
        data_in=df,
        user=user,
        method="dd0",
        params=params,
    )
    df_out = de.evaluate_detector()
    return params, df_out


def eval_dd1(df, user, params):
    de = evaluator.LeaveOneOutDetectorEvaluator(
        data_in=df,
        user=user,
        method="dd1",
        params=params,
    )
    df_out = de.evaluate_detector()
    return params, df_out


def dd0_runner():
    # Parameter 'template'
    dd0_template = {
        "precision": None,
        "theta": None,
    }

    # Parameter space. Parameter values to experiment with...
    precisions = [17, 18, 19]
    thetas = [0.05, 0.10, 0.20]
    dd0_experiments = []

    for par0 in precisions:
        for par1 in thetas:
            v = dd0_template.copy()
            v["precision"] = par0 # type: ignore # noqa
            v["theta"] = par1 # type: ignore # noqa
            dd0_experiments.append(v)

    arg_list = []
    for item in dd0_experiments:
        arg_list.append(["data/case_study_2.json", user, item])

    results_dd0 = []
    counter = 0
    for arg in arg_list:
        results_dd0.append(eval_dd0(*arg))
        # Dump data for each newly completed experiment
        pickle.dump(
            results_dd0,
            open(f"dd0_pickled_results_0_exp_{counter}.p", "wb"),
        )
        counter += 1

    # Print the results to stdout
    result_df = extract_stats(results_dd0, method = "dd0", aggregate = True)
    print("Results: ")
    print(result_df)
    
    # Save results
    pickle.dump(results_dd0, open("dd0_case_study_2_results_0.p", "wb"))


def dd1_runner():
    # Parameter 'template'
    dd1_template = {
        "precision": None,
        "threshold_low": 0.0,
        "threshold_high": None,
        "window_size": "1000T",
        "max_merging_distance": 28,
        "min_frequency": 1,
        "sequence_interpolator": ip.NaiveGeohashSequenceInterpolator,
    }

    # Parameter space. Parameter values to experiment with...
    precisions = [17, 18, 19]
    threshold_highs = [0.20, 0.40, 0.60]
    dd1_experiments = []

    for par0 in precisions:
        for par2 in threshold_highs:
            v = dd1_template.copy()
            v["precision"] = par0 # type: ignore # noqa
            v["threshold_high"] = par2 # type: ignore # noqa
            dd1_experiments.append(v)

    # Read in experimental data
    main_df = evaluator.read_all("data/case_study_2.json")

    arg_list = []
    for item in dd1_experiments:
        # arg_list.append(["data/case_study_2.json", user, "dd1", item])
        arg_list.append([main_df.copy(), user, item])

    results_dd1 = []
    counter = 0
    for arg in arg_list:
        results_dd1.append(eval_dd1(*arg))
        # Dump data for each newly completed experiment
        pickle.dump(
            results_dd1,
            open(f"dd1_pickled_results_0_exp_{counter}.p", "wb"),
        )
        counter += 1
    
    # Print the results to stdout
    result_df = extract_stats(results_dd1, method = "dd1", aggregate = True)
    print("Results: ")
    print(result_df)
    
    # Save results
    pickle.dump(results_dd1, open("dd1_case_study_2_results_0.p", "wb"))


if __name__ == "__main__":
    # dd0_runner()
    dd1_runner()
