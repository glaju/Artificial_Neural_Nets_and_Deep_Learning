T = [1 1; -1 -1; 1 -1]';
% plot(T(1,:),T(2,:),'r*')
% axis([-2 2 -2 2])
% title('Hopfield Network State Space')
% xlabel('a(1)');
% ylabel('a(2)');
net = newhop(T);
Ai = {[-0.1, 0.01]'};
Ai = {[-0.01, -0.2]'};
%[Y,Pf,Af] = sim(net,3,[],Ai);
Y = net({30},{},Ai);
Y{:}
%10, 40