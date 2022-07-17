% use
[~,nmi,~] = compute_nmi(gt,S);
acc = Accuracy(S,gt);
[f-score,p,r] = compute_f(gt,S);
[ar,~,~,~]=RandIndex(gt,S);