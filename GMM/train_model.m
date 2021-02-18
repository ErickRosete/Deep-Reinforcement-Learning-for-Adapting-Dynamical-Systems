function bll = train_model(directory, name, type, K, num_models)
    %% Load packages
    if isempty(strfind(path, "D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib"))
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib")
        addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\SEDS_lib")
    end
    %% Loading Data
    file_pattern = fullfile(directory, '*.txt');
    file_list = dir(file_pattern);
    for i = 1:length(file_list)
        file_name = fullfile(file_list(i).folder, file_list(i).name);
        data = dlmread([file_name]);
        if type == "force"
            x = data(:, 2:size(data, 2));
        else 
            x = data(:, 2:4);
        end
        demos{i} = x';
        t{i} = data(:,1)';
    end

    %% Preprocessing Data
    tol_cutting = .05; %.005 A threshold on velocity that will be used for trimming demos
    [x0 , xT, Data, index] = preprocess_demos(demos,t,tol_cutting); %preprocessing data

     %% GMM Fitting algorithm
     bll = 0;
     for i = 1:num_models
        [Priors_0, Mu_0, Sigma_0] = EM_init_kmeans(Data, K);
        [Priors_1, Mu_1, Sigma_1, nbStep, loglik] = EM(Data, Priors_0, Mu_0, Sigma_0);
        if loglik > bll
            bll = loglik;
            Priors = Priors_1; Mu = Mu_1; Sigma = Sigma_1;
        end
     end

    %% Exporting the trained model to the SEDS lib
    %out = export2SEDS_Cpp_lib(model_name, Priors, Mu, Sigma);
    model_name = strcat(name,'.mat');
    save(model_name, 'Priors', 'Mu', 'Sigma')
end

