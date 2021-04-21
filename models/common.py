# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

def get_current_num_jobs(it, num_it, start, step, end):
    "Get number of jobs for iteration number 'it' of range('num_it')"

    ideal = float(start) + (end - start) * float(it) / num_it
    if ideal < step:
        return int(0.5 + ideal)
    else:
        return int(0.5 + ideal / step) * step

def get_learning_rate(iter, num_jobs, num_iters, num_archives_processed,
                      num_archives_to_process,
                      initial_effective_lrate, final_effective_lrate):
    if iter + 1 >= num_iters:
        effective_learning_rate = final_effective_lrate
    else:
        effective_learning_rate = (
                initial_effective_lrate
                * math.exp(num_archives_processed
                           * math.log(float(final_effective_lrate) / initial_effective_lrate)
                           / num_archives_to_process))

    return num_jobs * effective_learning_rate

def get_successful_models(loss_list, difference_threshold=1.0):
    max_idx = loss_list.index(max(loss_list))
    accepted_models = []
    num_models = len(loss_list)
    for idx in range(num_models):
        if (loss_list[max_idx] - loss_list[idx]) <= difference_threshold: 
            accepted_models.append(idx)
        
    return accepted_models

