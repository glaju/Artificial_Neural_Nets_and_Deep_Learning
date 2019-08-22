clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
x=0:0.07:3*pi; y=sin(x.^2);
yn = y + 0.5* randn(1,numel(y));
p=con2seq(x); t=con2seq(y); tn=con2seq(yn);  % convert the data to a useful format

%creation of networks
net1=feedforwardnet(50,'trainlm');
net2=feedforwardnet(50,'trainlm');
net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=1;  % set the number of epochs for the training 
net2.trainParam.epochs=1;
net1=train(net1,p,t);   % train the networks
net2=train(net2,p,tn);
a11=sim(net1,p); a21=sim(net2,p);  % simulate the networks with the input vector p

net1.trainParam.epochs=14;
net2.trainParam.epochs=14;
net1=train(net1,p,t);
net2=train(net2,p,tn);
a12=sim(net1,p); a22=sim(net2,p);

net1.trainParam.epochs=985;
net2.trainParam.epochs=985;
[net1 tr1]=train(net1,p,t);
net2=train(net2,p,tn);
a13=sim(net1,p); a23=sim(net2,p);

err_lm =immse(y, cell2mat(a13));
err_br =immse(y, cell2mat(a23));

disp(err_lm)
disp(err_br)

%plots
figure
subplot(3,3,1);
plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','gradient descent','Levenberg-Marquardt','Location','north');
subplot(3,3,2);
postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
subplot(3,3,3);
postregm(cell2mat(a21),y);
%
subplot(3,3,4);
plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
title('15 epochs');
legend('target','gradient descent','Levenberg-Marquardt','Location','north');
subplot(3,3,5);
postregm(cell2mat(a12),y);
subplot(3,3,6);
postregm(cell2mat(a22),y);
%
subplot(3,3,7);
%plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
plot(x,y,'bx',x,yn,'rx',x,cell2mat(a23),'g');
title('1000 epochs');
legend('target','+noise','LM','Location','northeast');
subplot(3,3,8);
postregm(cell2mat(a13),y);
subplot(3,3,9);
postregm(cell2mat(a23),y);


figure(4)

%plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
plot(x,y,'bx',x,yn,'rx',x,cell2mat(a13),'m',x,cell2mat(a23),'g', 'linewidth', 2);
title('1000 epochs');
legend('target','+noise','LM','LM noisy','Location','northeast', 'fontsize',18);
ax = gca;
ax.FontSize = 18;
figure(5)
plotregression(cell2mat(a13), y, 'LM',cell2mat(a23), y, 'LM noisy')


k
%Create networks
net_GD=feedforwardnet(50,'traingd');
net_GDA=feedforwardnet(50,'traingda');
net_CGF=feedforwardnet(50,'traincgf');
net_CGP=feedforwardnet(50,'traincgp');
net_BFG=feedforwardnet(50,'trainbfg');
net_LM=feedforwardnet(50,'trainlm');

net_GDA.iw{1,1}=net_GD.iw{1,1};  %set the same weights and biases for the networks 
net_GDA.lw{2,1}=net_GD.lw{2,1};
net_GDA.b{1}=net_GD.b{1};
net_GDA.b{2}=net_GD.b{2};

net_CGF.iw{1,1}=net_GD.iw{1,1};  %set the same weights and biases for the networks 
net_CGF.lw{2,1}=net_GD.lw{2,1};
net_CGF.b{1}=net_GD.b{1};
net_CGF.b{2}=net_GD.b{2};

net_CGP.iw{1,1}=net_GD.iw{1,1};  %set the same weights and biases for the networks 
net_CGP.lw{2,1}=net_GD.lw{2,1};
net_CGP.b{1}=net_GD.b{1};
net_CGP.b{2}=net_GD.b{2};

net_BFG.iw{1,1}=net_GD.iw{1,1};  %set the same weights and biases for the networks 
net_BFG.lw{2,1}=net_GD.lw{2,1};
net_BFG.b{1}=net_GD.b{1};
net_BFG.b{2}=net_GD.b{2};

net_LM.iw{1,1}=net_GD.iw{1,1};  %set the same weights and biases for the networks 
net_LM.lw{2,1}=net_GD.lw{2,1};
net_LM.b{1}=net_GD.b{1};
net_LM.b{2}=net_GD.b{2};

net_GD.trainParam.epochs=500;  % set the number of epochs for the training 
net_GDA.trainParam.epochs=500;
net_CGF.trainParam.epochs=500; 
net_CGP.trainParam.epochs=500;
net_BFG.trainParam.epochs=500; 
net_LM.trainParam.epochs=500;

% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_GD,p,t);
% end
% elapsed_GD = toc(tstart)/10;
% fprintf('GD: %.2f s', elapsed_GD);
% 
% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_GDA,p,t);
% end
% elapsed_GDA = toc(tstart)/10;
% fprintf('GDA: %.2f s', elapsed_GDA);
% 
% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_CGF,p,t);
% end
% elapsed_CGF = toc(tstart)/10;
% fprintf('CGF: %.2f s', elapsed_CGF);
% 
% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_CGP,p,t);
% end
% elapsed_CGP = toc(tstart)/10;
% fprintf('CGP: %.2f s', elapsed_CGP);
% 
% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_BFG,p,t);
% end
% elapsed_BFG = toc(tstart)/10;
% fprintf('BFG: %.2f s', elapsed_BFG);
% 
% for i=1:11
%     if i == 2
%         tstart = tic;
%     end
%     train(net_LM,p,t);
% end
% elapsed_LM = toc(tstart)/10;
% fprintf('LM: %.2f s', elapsed_LM);


[net_GD, tr_GD]=train(net_GD,p,t);
[net_GDA, tr_GDA]=train(net_GDA,p,t);
[net_CGF, tr_CGF]=train(net_CGF,p,t);
[net_CGP, tr_CGP]=train(net_CGP,p,t);
[net_BFG, tr_BFG]=train(net_BFG,p,t);
[net_LM, tr_LM]=train(net_LM,p,t);
tr_GD.perf
figure(2)
plot(log10(tr_GD.perf),'y', 'LineWidth', 2,'DisplayName','GD')
hold on;
plot(log10(tr_GDA.perf),'m', 'LineWidth', 2,'DisplayName','GD adaptive LR')
plot(log10(tr_CGF.perf),'r', 'LineWidth', 2,'DisplayName','Fletcher-Reeves conjugate')

plot(log10(tr_CGP.perf),'g', 'LineWidth', 2,'DisplayName','Polak-Ribiere conjugate')
plot(log10(tr_BFG.perf),'b', 'LineWidth', 2,'DisplayName','BFGS quasi Newton')
plot(log10(tr_LM.perf),'c', 'LineWidth', 2,'DisplayName','Levenberg-Marquardt')
ytickformat('10^%i')
xlabel('Epoch', 'fontsize', 18)
ylabel('Mean Squared Error (mse)', 'fontsize', 18)
legend('fontsize', 16, 'location', 'southeast')
ax = gca;
ax.FontSize = 16; 
xlim([0 1000])
ylim([-10 3.2])



data_GD = sim(net_GD,p);
data_GDA = sim(net_GDA,p);
data_CGF = sim(net_CGF,p);
data_CGP = sim(net_CGP,p);
data_BFG = sim(net_BFG,p);
data_LM = sim(net_LM,p);
figure(3)
plotregression(cell2mat(data_GD), y, 'GD',cell2mat(data_GDA), y, 'GD adaptive LR',...
cell2mat(data_CGF), y, 'Fletcher-Reeves conjugate',...
cell2mat(data_CGP), y, 'Polak-Ribiere conjugate',...
cell2mat(data_BFG), y, 'BFGS quasi Newton',...
cell2mat(data_LM), y, 'Levenberg-Marquardt')

s_GD = 2.41/0.46;
a_GD = round(1:s_GD:501);
arr_GD = tr_GD.perf(a_GD);

s_LM =2.41/1.64;
a_LM = round(1:s_LM:501);
arr_LM = tr_LM.perf(a_LM);

s_GDA = 2.41/0.48;
a_GDA = round(1:s_GDA:501);
arr_GDA = tr_GDA.perf(a_GDA);

s_CGF = 2.41/1.03;
a_CGF = round(1:s_CGF:501);
arr_CGF = tr_CGF.perf(a_CGF);

s_CGP = 2.41/1.08;
a_CGP = round(1:s_CGP:501);
arr_CGP = tr_CGP.perf(a_CGP);

figure(8)
plot(log10(arr_GD),'y', 'LineWidth', 2,'DisplayName','GD')
hold on;
plot(log10(arr_GDA),'m', 'LineWidth', 2,'DisplayName','GD adaptive LR')
plot(log10(arr_CGF),'r', 'LineWidth', 2,'DisplayName','Fletcher-Reeves conjugate')

plot(log10(arr_CGP),'g', 'LineWidth', 2,'DisplayName','Polak-Ribiere conjugate')
plot(log10(tr_BFG.perf),'b', 'LineWidth', 2,'DisplayName','BFGS quasi Newton')
plot(log10(arr_LM),'c', 'LineWidth', 2,'DisplayName','Levenberg-Marquardt')

ytickformat('10^%i')
xlabel('time (s)', 'fontsize', 18)
ylabel('Mean Squared Error (mse)', 'fontsize', 18)
legend('fontsize', 16, 'location', 'northeast')
ax = gca;
ax.FontSize = 16; 
xlim([0 500])
ylim([-10 3.2])