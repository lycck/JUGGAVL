
clear;
clc;
omega = [0.01];
beta = [0.001];
lambda = [0.1];
gamma = [0.01];
knn = [15];

dataname = 'data/MSRCv1';

data = load(dataname);
X = data.X;
truth = data.Y;

str1='result.txt';
for i=1:length(omega)
    for j=1:length(beta)
        for ll=1:length(lambda)
            for l=1:length(gamma)
                for m=1:length(knn)
                    nmi = zeros(20,1);
                    pur = zeros(20,1);
                    acc = zeros(20,1);
                    ar = zeros(20,1);
                    f = zeros(20,1);
                    for nn = 1:20
                        [label,~,~,~] = model(X,truth,omega(i),beta(j),lambda(ll),gamma(l),knn(m));
                        [~,nmi(nn),~] = compute_nmi(label,truth);
                        [~,pur(nn)] = getFourMetrics(label, truth);
                        acc(nn) = Accuracy(truth,label);
                        [f(nn),p(nn),r(nn)] = compute_f(label,truth);
                        [ar(nn),~,~,~]=RandIndex(label,truth); 
                    end
                    dlmwrite(str1,[omega(i),beta(j),lambda(ll),gamma(l),knn(m),mean(nmi),std(nmi),mean(acc),std(acc),mean(pur),std(pur),mean(ar),std(ar),mean(f),std(f)],'-append','delimiter','\t','newline','pc');
                end
            end
        end
    end
end
