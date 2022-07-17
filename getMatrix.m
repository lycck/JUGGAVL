
clc;
clear;
omega = [0.001];
beta = [10];
lambda = [0.001];
gamma = [0.01];
knn = [15];

savePath = './result/';
dataname = {'newsgroup'};
for pp = 1:length(dataname)
    str1 = strcat(dataname{pp},'.mat');
    data = load(str1);
    X = data.X;
    truth = data.Y;
    for i=1:length(X)
       if size(X{i},2)~=length(truth)
          X{i}=X{i}'; 
       end
    end

path='./';
[label,A,Z,U] = model(X,truth,omega(pp),beta(pp),lambda(pp),gamma,knn);
[~,nmi,~] = compute_nmi(label,truth);
[~,pur] = getFourMetrics(label, truth);
acc = Accuracy(truth,label);
[f,p,r] = compute_f(label,truth);
[ar,~,~,~]=RandIndex(label,truth);

saveStr = strcat(dataname{pp}, '_matrixInfo.mat');
saveStr = strcat(savePath, saveStr);
save(saveStr,'nmi','acc','pur','ar','f','A','Z','U');
end
