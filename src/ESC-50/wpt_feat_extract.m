% This function takes in a given signal and extracts features using Multiscale Wavelet Packet Decomposition
% INPUTs : signal - signal vector to be decomposed 
% 		   n - number of levels of decomposition
% 		   label - class the input signal belongs to
% OUTPUTS : features - Lower and higher order statistical features extracted from given signal 
% AUTHOR : Kathikeyan Ganesan (karthikeyan@gatech.edu)
function features =  wpt_feat_extract(signal,n,label)
n_levels = n;
feature_mat = [];
for level = 1:n_levels
    t = wpdec(signal,level,'sym4');
    N = allnodes(t);
    terminal_nodes = N(2^level:2^(level+1)-1);
    last_level = read(t,'data',terminal_nodes);
    for j = 1:length(last_level)
        f = feature_extract(last_level{j});
        f = horzcat(f,label);
        feature_mat = [feature_mat;f];
    end      
end
features = feature_mat;
    
    
