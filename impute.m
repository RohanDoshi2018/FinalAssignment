data = csvread(data.csv);
data_filled = knnimpute(data');
data_filled = data_filled';
csvwrite('data_imputed.csv',x)