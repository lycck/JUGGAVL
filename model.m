function [label,A,Z,U]=model(X,truth,omega,beta,lambda,gamma,knn)

%normalize
for i=1:length(X)
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1));
end

n = size(X{1},2);
tol = 1e-8; tol2 = 1e-6;
iter = 0;
obj_change = inf;
maxiter = 10;
obj = zeros(maxiter,1);
changes = zeros(maxiter,1);
v = length(X);
true_Z = cell(v,1);
W = cell(1,v);
E = cell(1,v);
c = length(unique(truth));
Lw = cell(1,v);

for i=1:v
    distance_matrix = pdist2(X{i}', X{i}', 'squaredeuclidean');
    sigma = mean(mean(distance_matrix));
    W{i} = exp(-distance_matrix/(2*sigma));
    [W{i}, ~] = kNN(W{i}, knn);
    Lw{i} = diag(sum(W{i})) - W{i};
end


%initialize
Z = cell(1,v);
for i=1:v
    E{i} = zeros(size(X{i}));
    A_syl = omega*(X{i}'*X{i});
    B_syl = beta*Lw{i};
    C_syl = omega*(-X{i}'*X{i}+X{i}'*E{i});
    tmp = lyap(A_syl,B_syl,C_syl);
    tmp(tmp<0) = 0;
    Z{i} = tmp;
end
alpha = ones(v,1);
D = cell(1,v);
S = Z{1};
A = cell(1,v);
for i=1:v
   A{i} = Z{i}; 
end


warning('off','MATLAB:nearlySingularMatrix')

while iter < maxiter && obj_change > tol
    
    iter = iter + 1;
    %fix Z,A,update alpha,S
    oldalpha = alpha;
    oldS = S;
    obj1 = 0;
    for i=1:v
        D{i} = Z{i} - A{i};
        temp = norm(alpha(i)*A{i}-S, 'fro');
        obj1 = obj1 + temp*temp;
    end
    T = zeros(v);
    R = zeros(v);
    for i=1:v
        for j=1:v
            T(i,j) = trace(A{i}*A{j}');
            T(j,i) = T(i,j);
            if i~=j
                R(i,j) = lambda*trace(D{i}*D{j}');
                R(j,i) = R(i,j);
            end
        end
    end
    K = 2*((eye(v) - (1/v)*ones(v)).* T + R);
    ONE = ones(v, 1);
    
    obj2 = sum(sum(R .* (alpha * alpha')));
    
    tmp = [K,ONE;ONE',0];
    tmpb = [zeros(v,1);1];
    solution = (tmp + 1e-8*eye(v+1)) \ tmpb;  %防止不可逆
    alpha = EProjSimplex_new(solution(1:v));
    
    S = zeros(n);
    for i=1:v
       S = S + alpha(i)*A{i}; 
    end
    S = S/v;
    
    alpha_change = norm(alpha-oldalpha, 'fro');
    Schange = norm(S-oldS, 'fro');
    
    
    %fix S,alpha,Z,update A
    alp_coef = alpha * alpha';
    coef = alp_coef .*  (lambda * ones(v) - lambda * diag(ones(1,v)) + diag(ones(1,v)));
    comZ = zeros(n);
    for i=1:v
       comZ = comZ + lambda*alpha(i)*Z{i};
    end
    for i=1:v
       tmpp = alpha(i) * (comZ - lambda*alpha(i)*Z{i}) + alpha(i) * S;  
       true_Z{i} = tmpp(:);       
    end
    tempb = cat(2,true_Z{:})';
    coef = coef + 1e-5*eye(size(coef));  %防止不可逆
    solution = (coef\tempb)';
    solution(solution<0) = 0;
    A_change = 0;
    for i=1:v
        temp = solution(:,i);
        oldA = A{i};
        A{i} = reshape(temp,n,n);
        A{i} = max(A{i}, A{i}');
        A{i} = min(Z{i}, A{i});
        A_change = A_change + norm(oldA-A{i}, 'fro');
    end
    
    
    %update Z
    Z_change = 0;
    sumI = zeros(n);
    comA = zeros(n);
    trueI = cell(1,v);
    trueA = cell(1,v);
    obj3 = 0;
    for i=1:v
       obj3 = obj3 + omega*norm(X{i}-X{i}*Z{i}-E{i},'fro')^2 + beta*trace(Z{i}*Lw{i}*Z{i}'); 
       sumI = sumI + lambda * alpha(i) * eye(n);
       comA = comA + lambda * alpha(i)*A{i};
    end
    for i=1:v
       trueI{i} = sumI - lambda * alpha(i) * eye(n);
       trueI{i} = alpha(i)*trueI{i};
       trueA{i} = alpha(i)*(comA - lambda * alpha(i)*A{i});
    end
    for i=1:v
        oldZ = Z{i};
        A_syl = omega*X{i}'*X{i} + trueI{i};
        B_syl = beta*Lw{i};
        C_syl = -(trueA{i} + omega*X{i}'*X{i} - omega*X{i}'*E{i});
        tmp = lyap(A_syl,B_syl,C_syl);
        tmp(tmp<0) = 0;
        Z{i} = tmp;
        Z_change = Z_change + norm(Z{i} - oldZ,'fro');
    end
    
    
    %Update E
    obj4 = 0;
    E_change = 0;
    for i=1:v
        oldE = E{i};
        tmpp1 = 0.5*gamma/omega;
        tmpp2 = X{i} - X{i}*Z{i};
        obj4 = obj4 + gamma * norm(E{i},1);
        oldE = oldE + E{i};
        E{i} = max(0,tmpp2-tmpp1) + min(0,tmpp2+tmpp1);
        E_change = E_change + norm(E{i} - oldE,'fro');
    end
     
        
    obj(iter) = obj1 + obj2 + obj3 + obj4;
%     obj(iter)
    change = A_change + alpha_change + Schange + Z_change + E_change;
    if iter > 1
        obj_change = min(abs(obj(iter)-obj(1:iter-1)))/abs(obj(1) - obj(iter));
    end
    changes(iter) = change;
    if iter > 20
        tol = tol2;
    elseif iter > 30
        tol = tol2*10;
    end
end

warning('on','MATLAB:nearlySingularMatrix')

S = S - diag(diag(S));
U = kNN(S,knn);
[label] = SpectralClustering(U, c, 3); 

