#!/usr/bin/env python

import os, sys
import argparse
sys.path.append(os.path.join(os.getcwd(), 'x64', 'Release'))

import swaption_pricing
import numpy as np
import timeit
from matplotlib import pyplot as plt

def approximate(w, x, y):
    return y

def swaption_exposure(swap_exposure, sdf, ex_indices, maturity_index):
    assert len(swap_exposure) == len(sdf)
    assert len(ex_indices) > 0
    prematurity_index = maturity_index - 1
    swaption_exposure = np.zeros_like(swap_exposure)
    is_exercise = np.zeros((len(ex_indices), swaption_exposure.shape[1]), dtype=np.bool_)
    current_option_slice = swaption_exposure[0].copy()
    current_ex = len(ex_indices) - 1
    for step in range(prematurity_index, -1, -1):
        if step != prematurity_index:
            discounted_opt_price = current_option_slice * sdf[step + 1] / sdf[step]
            current_option_slice = approximate(sdf[step], swap_exposure[step], discounted_opt_price)
        if current_ex >= 0 and step == ex_indices[current_ex]:
            is_exercise[current_ex] = current_option_slice < swap_exposure[step]
            current_option_slice = np.maximum(current_option_slice, swap_exposure[step])
            while current_ex >= 0 and step == ex_indices[current_ex]:
                current_ex -= 1
        swaption_exposure[step] = current_option_slice
    for i in range(1, len(ex_indices)):
        is_exercise[i] |= is_exercise[i - 1]
    interval_endings = [ex_index + 1 for ex_index in ex_indices[1:]] + [len(swap_exposure)]
    for left_index, right_index, is_exercise_slice in zip(ex_indices, interval_endings, is_exercise):
        swaption_exposure[left_index + 1:right_index, is_exercise_slice] = swap_exposure[left_index + 1:right_index, is_exercise_slice]
    return np.mean(swaption_exposure[0]), swaption_exposure

def swaption_exposure_noreturn(swap_exposure, sdf, ex_indices, maturity_index):
    swaption_exposure(swap_exposure, sdf, ex_indices, maturity_index)


def measure(stmt, globals):
    results = timeit.repeat(stmt, repeat=7, number=100, globals=globals)
    print(stmt)
    print(f"{np.mean(results)} ± {np.std(results)}")
    print()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=int, default=1000)
    parser.add_argument('--grid', type=int, default=100)
    parser.add_argument('--ex_times', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(sys.argv[1:])
    inputs = swaption_pricing.setup(args.paths, args.grid, args.ex_times, args.seed)
    swap_exposure, sdf, ex_indices, maturity_index = inputs.swap_exposure, inputs.sdf, inputs.ex_indices, inputs.maturity_index 
    swap_exposure, sdf = map(np.array, (swap_exposure, sdf))
    
    # run exposures both versions
    npv1, sopt_exp1 = swaption_exposure(swap_exposure, sdf, ex_indices, maturity_index)
    outputs2 = swaption_pricing.swaption_exposure(inputs)
    npv2, sopt_exp2 = outputs2.npv, outputs2.swaption_exposure
    print("Differences between Py and Cpp:", np.abs(npv1 - npv2), np.abs(sopt_exp1 - sopt_exp2).max())

    measure('npv1, sopt_exp1 = swaption_exposure(swap_exposure, sdf, ex_indices, maturity_index)', globals=locals())
    measure('outputs2 = swaption_pricing.swaption_exposure(inputs)', globals=locals())




