function xplus = EvolveState(A, B, K, x0, w)

xplus = [];

for i = 1:size(x0, 1)
    x0i = x0(i, :)';
    wi = w(i, :)';
    u = K * x0i;
    x = A * x0i + B * (sign(u) .* min(abs(u), 1)) + wi;
    
    xplus = [xplus ; x'];
end