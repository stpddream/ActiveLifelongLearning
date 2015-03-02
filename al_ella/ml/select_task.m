function [ task_id ] = select_task(model, seed_dat, selectionCriterion)
%SELECT_TASK Summary of this function goes here
%   Detailed explanation goes here
    Xs = seed_dat.feature;
    Ys = seed_dat.label;
    task_id = selectTaskELLA(model, Xs, Ys, selectionCriterion);  
end

