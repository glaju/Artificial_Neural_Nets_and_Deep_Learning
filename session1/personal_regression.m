clear
clc
close all
load('Data_Problem1_regression.mat')
Tnew = (7*T1 + 6*T2 + 6*T3 + 6*T4 + 5*T5)/(7 + 6 + 6 + 6 + 5);
r = randi([1 13600],1,1000);

X1_train = X1(r);
X2_train = X2(r);
Tnew_train = Tnew(r);
r = randi([1 13600],1,1000);

X1_val = X1(r);
X2_val = X2(r);
Tnew_val = Tnew(r);
r = randi([1 13600],1,1000);
X1_test = X1(r);
X2_test = X2(r);
Tnew_test = Tnew(r);

% figure(1)
% dt = delaunayTriangulation(X1_train,X2_train) ;
% tri = dt.ConnectivityList ;
% trisurf(tri,X1_train,X2_train,Tnew_train)
X_train = [X1_train'; X2_train'];
size(X_train)
X_val = [X1_val'; X2_val'];
X_test = [X1_test'; X2_test'];
% net = feedforwardnet(10);
% 
% net.inputConnect = [1 1; 1 1];
% net = configure(net,X_train);
% disp('start')
% net = train(net,X_train, Tnew_train);
% disp('done')
% 
% %view(net)
% y = net(X_train);
% %perf = perform(net,X_train, Tnew_train)

net = feedforwardnet([5,5,5],'trainlm');
net.trainParam.epochs=1000;
% net.divideParam.trainRatio=1;
% net.divideParam.testRatio=0;
% net.divideParam.valRatio=0;
%net.numinputs =2;
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';
disp(size(Tnew_train'))
disp(size(X_train))
net = train(net, X_train, Tnew_train');
data_val = sim(net,X_val);
figure(3)
plotregression(data_val, Tnew_val')

data_test= sim(net,X_test);
data_train= sim(net,X_train);
% figure(1)
% dt = delaunayTriangulation(X1_test,X2_test) ;
% tri = dt.ConnectivityList ;
% trisurf(tri,X1_test,X2_test,Tnew_test)
% 
% figure(2)
% dt = delaunayTriangulation(X1_test,X2_test) ;
% tri = dt.ConnectivityList ;
% trisurf(tri,X1_test,X2_test,data_test')

err_train =immse(data_train, Tnew_train');
err_val =immse(data_val, Tnew_val');
err_test = immse(data_test, Tnew_test');
disp(err_train)
disp(err_val)
disp(err_test)