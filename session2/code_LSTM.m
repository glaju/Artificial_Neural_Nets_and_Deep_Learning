clear all;
dataTrain = load('./lasertrain.dat');
dataTrain = dataTrain';
dataTest = load('./laserpred.dat');
dataTest = dataTest';

% figure
% plot(data)
% xlabel("discrete time k")
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

dataTestStandardized = (dataTest - mu) / sig;

XTest = dataTestStandardized(1:end-1);
YTest = dataTestStandardized(2:end);

%net = predictAndUpdateState(net,XTrain);
%net = resetState(net);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,XTest(1));
YPred
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end


net = resetState(net);
[net,YPred_tr] = predictAndUpdateState(net,XTrain(1));

numTimeStepsTest = numel(XTrain);
for i = 2:numTimeStepsTest
    [net,YPred_tr(:,i)] = predictAndUpdateState(net,YPred_tr(:,i-1),'ExecutionEnvironment','cpu');
end
YPred_tr = sig*YPred_tr + mu;
YTrain = sig*YTrain + mu;
YPred = sig*YPred + mu;
YTest = sig*YTest + mu;


rmse = sqrt(mean((YPred-YTest).^2))
rmse2 = sqrt(mean((YPred_tr-YTrain).^2))
numTimeStepsTrain=999;
% figure
% plot(dataTrain(1:end-1))
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% hold off
% xlabel("Month")
% ylabel("Cases")
% title("Forecast")
% legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTrain)
hold on
plot(YPred_tr,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred_tr - YTrain)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse2)

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse2)