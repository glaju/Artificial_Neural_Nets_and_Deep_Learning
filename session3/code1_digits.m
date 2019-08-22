
% pbaspect([1 1 1])
figure(1)
colormap('gray')
set(gca,'xtick',[],'ytick',[]);
imagesc(reshape(threes(1,:),16,16),[0,1])
imagesc(reshape(mean(threes(:,:)),16,16),[0,1])
axis off;
pbaspect([1 1 1])

cov_m = cov(threes);
[v,d] = eig(cov_m);
figure(6)
%plot(diag(d), 'linewidth', 2)
scatter(1:256, diag(d), 30, 'bo', 'MarkerFaceColor', 'b')
ylabel('eigenvalue', 'fontsize',30)
set(gca,'FontSize',30);

[v1,d1]=eigs(cov_m,1);
[v2,d2]=eigs(cov_m,2);
[v3,d3]=eigs(cov_m,3);
[v4,d4]=eigs(cov_m,4);

[residuals1,reconstructed1] = pcares(threes,1);
[residuals2,reconstructed2] = pcares(threes,2);
[residuals3,reconstructed3] = pcares(threes,3);
[residuals4,reconstructed4] = pcares(threes,4);
figure(2)
colormap('gray')
subplot(151);

imagesc(reshape(threes(1,:),16,16),[0,1])
title('Original', 'Fontsize', 20)
pbaspect([1 1 1])
axis off;
subplot(152);

imagesc(reshape(reconstructed1(1,:),16,16),[0,1])
title('1 eigenvalue', 'Fontsize', 20)
pbaspect([1 1 1])
axis off;
subplot(153);

imagesc(reshape(reconstructed2(1,:),16,16),[0,1])
title('2 eigenvalues', 'Fontsize', 20)
axis off
pbaspect([1 1 1])
subplot(154);

imagesc(reshape(reconstructed3(1,:),16,16),[0,1])
title('3 eigenvalues', 'Fontsize', 20)
axis off
pbaspect([1 1 1])
subplot(155);

imagesc(reshape(reconstructed4(1,:),16,16),[0,1])
title('4 eigenvalues', 'Fontsize', 20)
axis off
pbaspect([1 1 1])

errors = zeros(1, 50);
for k=1:50
    errors(k) = compute_error(threes, k);
end



dia = fliplr(diag(d)');
missing = zeros(1,50);
for i=1:50
    
    missing(i) = sum(dia(i+1:end));
end
% missing = fliplr(missing);
figure(3)
hold on;
subplot(121)
plot(errors, 'b', 'linewidth', 2);
ylabel('Error using k eigenvalues', 'Fontsize', 20)
xlabel('k','Fontsize', 20)
set(gca,'FontSize',20);
subplot(122)
plot(missing, 'r', 'linewidth', 2)
ylabel('Sum of eigenvalues not in use','Fontsize', 20);
xlabel('k','Fontsize', 20)
set(gca,'FontSize',20);

figure(4)
scatter(missing, errors,20, 'bo', 'MarkerFaceColor', 'b')
xlabel('Sum of eigenvalues not in use','Fontsize', 20);
ylabel('Error using k eigenvalues', 'Fontsize', 20)
set(gca,'FontSize',20);