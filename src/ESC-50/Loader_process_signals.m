%This file loads the individual subjects' EEG data and preprocesses them using a multiscale PCA algorithm before performing wavelet packet decomposition 
%on the signal
%Author : Karthikeyan Ganesan (karthikeyan@gatech.edu)
% Modified on : 17th Nov 2019

%% load file in individual file directory
clear all; close all;
parent_dir = 'G:\DDL_Project\src\ESC-50\class_mat'; 
base = dir(parent_dir);
tot_levels = 5;
for i = 3:length(base)
    if(base(i).isdir)
%         fprintf("Processing subject %s trials.....\n",base(i).name);
        feature_class = [];
        sub_dir_name = fullfile(parent_dir,base(i).name);
        local_files = dir(sub_dir_name);
        for j = 3:length(local_files)
%             fprintf("Processing %d file of subject %s.....\n",j,base(i).name);
            [signal,fs] = audioread(local_files(j).name);
            %%%% Denoise signal with MultiScale PCA %%%%
            fprintf("Denoising with MPCA \n");
            [x_denoise,~,pc,~,pca_params] = wmspca(signal,5,'sym4','kais');
            [b,a] = size(signal);
            label = str2double(extractAfter(base(i).name,'class'));
            %%%% For each channel do WPT and extract features %%%%
            fprintf(" Extracting features for denoised signal \n");
            wpt_features = getmswpfeat(signal,fs,fs,tot_levels,'matlab');
            coeff = mfcc(signal,fs,'WindowLength',fs,'OverlapLength',0);
            local_lpc = zeros(1,10);
            for win = 1:fs:length(signal)-fs
                [lpc_feature,~] = lpc(signal(win:win+fs),9);
                local_lpc = [local_lpc;lpc_feature];
            end 
            fprintf("LPC feature extraction complete \n");
            file_features = horzcat(wpt_features,coeff,local_lpc);
            file_mfcc_features = mean(coeff,1);
            file_features = mean(file_features,1);
            feature_class = [feature_class;file_features];
            fprintf("Feature extracted for %d file of class %d \n",j,i);
            feature_file_name = fullfile(parent_dir,base(i).name,local_files(j).name);
            feature_file_name = feature_file_name(1:end-4);
            save(feature_file_name,'file_features','file_mfcc_features','label');
        end 
    end 
    global_features{i} = feature_class;
end 
                
                