function feat_vect  = feature_extract(vector)
signal_mean = mean(vector);
signal_power = sum(vector.^2)/length(vector);
signal_dev = std(vector);
signal_hos_1 = skewness(vector);
signal_hos_2 = kurtosis(vector);
feat_vect = horzcat(signal_mean,signal_power,signal_dev,signal_hos_1,signal_hos_2);

