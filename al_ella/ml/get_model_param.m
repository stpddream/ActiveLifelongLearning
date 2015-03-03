function [ params ] = get_model_param(L, S, t)
    params = dot(L, S(:, t));
end

