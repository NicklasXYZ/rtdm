import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from stream_handler import setup
setup()
import pickle

import backend.detectorevaluator as deteval
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


# extract_stats(results_dd0, method = "dd0", aggregate = True)
# extract_stats(results_dd1, method = "dd1", aggregate = True)


def eval_dd0(json_file, user, method, params):
    de = deteval.LeaveOneOutDetectorEvaluator(
        json_file="data/case_study_2.json",
        user=user,
        method="dd0",
        params=params,
    )
    df_out = de.evaluate_detector()
    return params, df_out


def eval_dd1(json_file, user, method, params):
    de = deteval.LeaveOneOutDetectorEvaluator(
        json_file="data/case_study_2.json",
        user=user,
        method="dd1",
        params=params,
    )
    df_out = de.evaluate_detector()
    return params, df_out


def dd0_runner():
    ########
    dd0_template = {
        "precision": None,
        "theta": None,
    }

    # Parameter values
    precisions = [17, 18, 19]
    thetas = [0.05, 0.10, 0.20]
    dd0_experiments = []

    for par0 in precisions:
        for par1 in thetas:
            v = dd0_template.copy()
            v["precision"] = par0
            v["theta"] = par1
            dd0_experiments.append(v)

    arg_list = []
    for item in dd0_experiments:
        arg_list.append(["data/case_study_2.json", user, "dd0", item])

    results_dd0 = []
    counter = 0
    for arg in arg_list:
        results_dd0.append(eval_dd0(*arg))
        # Dump data for each newly completed experiment
        pickle.dump(
            results_dd0,
            open(f"dd0_pickled_results_large_0_exp_{counter}.p", "wb"),
        )
        counter += 1

    # with multiprocessing.Pool(processes = 10) as p:
    #     results_dd0 = p.starmap(
    #         eval_dd0,
    #         arg_list,
    #     )

    for item in results_dd0:
        print("Result: ")
        print(item)
        print()

    pickle.dump(results_dd0, open("dd0_pickled_results_large_0.p", "wb"))


def dd1_runner():
    ########
    dd1_template = {
        "precision": None,
        "threshold_low": 0,
        "threshold_high": None,
        "window_size": "1000T",
        "max_merging_distance": 28,
        "min_frequency": 1,
        "sequence_interpolator": ip.NaiveGeohashSequenceInterpolator,
    }

    # Parameter values
    precisions = [17, 18, 19]
    threshold_highs = [0.20, 0.40, 0.60]
    dd1_experiments = []

    # for par0 in precisions:
    #     for par1 in window_sizes1:
    #         for par2 in threshold_highs:
    #             v = dd1_template.copy()
    #             v["precision"] = par0
    #             v["window_size1"] = par1
    #             v["threshold_high"] = par2
    #             dd1_experiments.append(v)

    for par0 in precisions:
        for par2 in threshold_highs:
            v = dd1_template.copy()
            v["precision"] = par0
            v["threshold_high"] = par2
            dd1_experiments.append(v)

    arg_list = []
    for item in dd1_experiments:
        arg_list.append(["data/case_study_2.json", user, "dd1", item])

    results_dd1 = []
    counter = 0
    for arg in arg_list:
        results_dd1.append(eval_dd1(*arg))
        # Dump data for each newly completed experiment
        pickle.dump(
            results_dd1,
            open(f"dd1_pickled_results_large_0_exp_{counter}.p", "wb"),
        )
        counter += 1

    # with multiprocessing.Pool(processes = 10) as p:
    #     results_dd1 = p.starmap(
    #         eval_dd1,
    #         arg_list,
    #     )

    for item in results_dd1:
        print("Result: ")
        print(item)
        print()

    pickle.dump(results_dd1, open("dd1_pickled_results_large_0.p", "wb"))


if __name__ == "__main__":
    # dd0_runner()
    dd1_runner()
