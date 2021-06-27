function [Priors, Mu, Sigma] = get_model(model_name)
    load(model_name,'Priors','Mu','Sigma') % loading the model
end