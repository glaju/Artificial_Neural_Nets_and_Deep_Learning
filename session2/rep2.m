%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=20;
for i=1:n
    a={rands(2,1)}; 
    
    % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps              
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r', 'linewidth', 2); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO', 'linewidth', 2);  % plot the final point with a green circle
end
%legend('initial state','time evolution','attractor','Location', 'northeast', 'fontsize', 20);
axis([-1.1 1.1 -1.1 1.1])
set(gca,'FontSize',20)
set(gca,'DataAspectRatio',[1 1 1])
title('2D Hopfield model - time evolution');
