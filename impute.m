data = csvread(data_processed.csv);
data_imputed = knnimpute(data');
data_imputed = data_imputed';
csvwrite('data_imputed.csv',data_imputed)