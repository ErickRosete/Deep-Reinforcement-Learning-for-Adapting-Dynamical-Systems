if isempty(strfind(path, "D:\Freiburg\Master_project\Deep-Reinforcement-Learning-for-Adapting-Dynamical-Systems"))
    addpath("D:\Freiburg\Master_project\Deep-Reinforcement-Learning-for-Adapting-Dynamical-Systems")
end    
if isempty(strfind(path, "D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib"))
    addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\GMR_lib")
    addpath("D:\Freiburg\Master_project\haptics_imitation\ds-opt\seds\SEDS_lib")
end
%% Loading Data
figure
hold on;grid on
clear demos t
num_demo = 0;
type = "pose";
file_pattern = fullfile("demonstrations_txt", '*.txt');
file_list = dir(file_pattern);
for i = 1:length(file_list)
    file_name = fullfile(file_list(i).folder, file_list(i).name);
    data = dlmread([file_name]);
    if type == "force"
        x = data(:, 2:size(data, 2));
    else 
        x = data(:, 2:4);
    end
    plot3(x(:,1),x(:,2),x(:,3),'r.')
    demos{i} = x';
    t{i} = data(:,1)';
end

%% Preprocessing Data
tol_cutting = .05; %.005 A threshold on velocity that will be used for trimming demos
[x0 , xT, Data, index] = preprocess_demos(demos,t,tol_cutting); %preprocessing data

figure
hold on;grid on
% cx = cos(Data(5,:)).*cos(Data(6,:));
% cy = sin(Data(5,:)).*cos(Data(6,:));
% cz = sin(Data(6,:));
% quiver3(Data(1,:),Data(2,:),Data(3,:), cx, cy, cz)
plot3(Data(1,:),Data(2,:),Data(3,:),'r.')
d = size(Data,1)/2;
xlabel('x')
ylabel('y')
zlabel('z')
view(-44,28)
title('Demonstrations after the preprocessing step.')
axis equal
pause(0.1)