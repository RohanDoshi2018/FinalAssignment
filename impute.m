data = csvread('data_super_nan.csv',1,1);
data_imputed = knnimpute(data');
data_imputed = data_imputed';
csvwrite('data_imputed.csv',data_imputed)