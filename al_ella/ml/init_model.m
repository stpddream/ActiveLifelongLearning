function model = init_model(init_dat)
%INIT_MODEL Summary of this function goes here
%   Detailed explanation goes here
    
    d = 20;
    useLogistic = true;
    T = length(init_dat.feature);
	d = size(init_dat.feature{1},2);
    
    model = initModelELLA(struct('k',2,...
	    			     'd',d,...
	    			     'mu',exp(-12),...
	    			     'lambda',exp(-10),...
	    			     'ridgeTerm',exp(-5),...
	    			     'initializeWithFirstKTasks',true,...
	    			     'useLogistic',useLogistic,...
	    			     'lastFeatureIsABiasTerm',true));
    X = init_dat.feature;
    Y = init_dat.label;
  
    learned = logical(zeros(length(Y),1));
	unlearned = find(~learned);
    
    % Use Initial task to initialize Model 
	for t = 1 : T
        
	    % change the last input to 1 for random, 2 for InfoMax, 3 for Diversity, 4 for Diversity++
	    % idx = selectTaskELLA(model,{X{unlearned}},{Y{unlearned}},2);
        idx = t; % Use sequential task adding first
	    model = addTaskELLA(model,X{idx}, Y{idx}, idx);
	    %learned(unlearned(idx)) = true;
	    %unlearned = find(~learned);
    end
end

