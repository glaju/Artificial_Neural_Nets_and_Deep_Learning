clear all;
p = 5;
N = 8;

train_data=load('./lasertrain.dat');
% ma = max(train_data);
% mi=min(train_data);
% for i=1:1000
%     train_data(i) = (train_data(i)-mi)/(ma-mi);
% end
% 
test_data = load('./laserpred.dat');
% ma = max(test_data);
% mi=min(test_data);
% for i=1:100
%    test_data(i) = (test_data(i)-mi)/(ma-mi);
% end

target = train_data(p+1:end)';
train_data = getTimeSeriesTrainData(train_data, p);
size(train_data)

target_test = test_data(p+1:end)';
%test_data = getTimeSeriesTrainData(test_data, p);



net=feedforwardnet(N,'trainlm');

net.divideParam.trainRatio=1;
net.divideParam.testRatio=0;
net.divideParam.valRatio=0;
net.trainParam.epochs=1000;
net = train(net, train_data, target);

data_pred_tr = sim(net,train_data);

size(data_pred_tr)
for i=1:100
    elem = sim(net, data_pred_tr(end-4:end)');
    data_pred_tr = [data_pred_tr elem];
end
size(data_pred_tr)



size(test_data)
test_data=test_data';
size(data_pred_tr(996:end))
err_test = immse(test_data, data_pred_tr(996:end));
disp(err_test)
rmse = sqrt(mean((test_data-data_pred_tr(996:end)).^2))
figure(3)

plotregression(test_data, data_pred_tr(996:end),'test')

figure(4)
subplot(2,1,1)
plot(test_data)
hold on
plot(data_pred_tr(996:end),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(data_pred_tr(996:end) - test_data)
xlabel("Month")
ylabel("Error")
title("RMSE" )
