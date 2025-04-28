% Carlo Karam, Matteo Tacchi, Mirko Fiacchini

clc, clear, close all

samples = 1e3;
horizon = 100;

%% System Defintion

A = [0.89, 0.1; 0.1, 0.89];
B = [0; 1];

n = size(A, 1);
m = size(B, 2);

noise_var = 1;
W = noise_var * eye(n);

w = mvnrnd(zeros(1, n), W, samples*horizon);

% Symmetric saturation bound
u_bound = 10;

K = [-0.2820, -0.8415];
%% OL Contraction LMI

P = sdpvar(n, n, 'symmetric');

% Bisection on lambda
lambda_min = 0;
lambda_max = 0.99;
lambda = (lambda_max + lambda_min)/2;

while (lambda_max - lambda_min) > 1e-4
    
    F = [
        P >= eye(n),...
        A' * P * A <= lambda * P,...
        (A + B*K)' * P * (A + B*K) <= lambda * P,...
    ];

    diagnostics = optimize(F, 0);

    if diagnostics.problem == 1
        lambda_min = lambda;
    else
        lambda_max = lambda;
    end

    lambda = (lambda_max + lambda_min)/2;
end

P = value(P);
lambda = lambda_max;

%% CL Contraction LMI

lambda_l = sdpvar(1, 1);

schur_comp = [lambda_l * P, (A+B*K)'; (A+B*K), inv(P)];

F = [lambda_l >= 0, schur_comp >= 0];
optimize(F, lambda_l);

lambda_l = value(lambda_l);

%% Compute lambda_bar, effective contraction rate

r_L = (u_bound^2) ./ (K*inv(P)*K');

if r_L > trace(P*W) * 1/(1- lambda)
    lambda_bar = sdpvar(1, 1);
    F = [
        trace(P*W)*(1)/(1 - lambda_bar) == r_L * (lambda_bar - lambda_l)/(lambda - lambda_l),... 
        lambda_bar <= lambda,...
    ];
    
    optimize(F, lambda_bar);
    lambda_bar = value(lambda_bar);
else
    lambda_bar = lambda;
end

%% Generate Samples with computed K
x0 = zeros(samples, n);
X = x0;
x = x0;

for i = 1:horizon
    wi = w((i-1)*samples+1:i*samples, :);
    x = EvolveState(A, B, K, x, wi,u_bound);
    X = [X; x];
end

%% Construct Expectation Bounds

sampled_pnorms = [];
sampled_mean = [0];
lambda_bound = [0];
lambda_l_bound = [0];
lambda_bar_bound = [0];

for i = 1:samples*(horizon+1)
    sampled_pnorms = [sampled_pnorms; X(i, :)*P*X(i, :)'];
end

for i = 1:horizon
    sampled_mean = [sampled_mean; mean(sampled_pnorms((i)*samples+1:(i+1)*samples))];
    lambda_bound = [lambda_bound; trace(P*W) * (1-lambda^i)/(1 - lambda)];
    lambda_l_bound = [lambda_l_bound; trace(P*W) * (1-lambda_l^i)/(1 - lambda_l)];
    lambda_bar_bound = [lambda_bar_bound; trace(P*W) * (1-lambda_bar^i)/(1 - lambda_bar)];
end


figure(1)
hold on;
grid on;

plot(sampled_mean, 'Color', 'red', 'Marker', '*');
plot(lambda_bound, 'Color', 'black', 'LineStyle', '--');
plot(lambda_l_bound, 'Color', 'green', 'LineStyle', '--');
plot(lambda_bar_bound, 'Color', 'blue', 'LineStyle', '--');

%% Visualize PUB

violation_prob = 0.2;

lambda_pub_radius =  (1/(violation_prob * (1 - lambda))) * trace(P*W);
lambda_bar_pub_radius = (1/(violation_prob * (1 - lambda_bar))) * trace(P*W);


figure(2)
hold on;

plot(X(horizon*samples:end, 1), X(horizon*samples:end, 2), '.r')

ee = sdpvar(n, 1);
plot(ee'*P*ee <= lambda_pub_radius,ee,[],[],sdpsettings('plot.shade',0, 'plot.linewidth', 2, 'plot.wirestyle', '-', 'plot.edgecolor', 'k'));
plot(ee'*P*ee <= lambda_bar_pub_radius,ee,[],[],sdpsettings('plot.shade',0, 'plot.linewidth', 2, 'plot.wirestyle', '-', 'plot.edgecolor', 'b'));

%% Viusalize lambda_bar evolution

r_Ls = [];
sat_bounds = 8.6:2:40;
lambdas_bar = [];

for i = 1:length(sat_bounds)
    radius_lin = sat_bounds(i)^2 ./ (K*inv(P)*K');
    r_Ls = [r_Ls; radius_lin];
    lambda_b = sdpvar(1,1);
    F = [trace(P*W)*(1)/(1 - lambda_b) == radius_lin * (lambda_b - lambda_l)/(lambda - lambda_l), lambda_b <= lambda];
    optimize(F, lambda_b);
    lambdas_bar = [lambdas_bar; value(lambda_b)];
end

figure(3)
hold on;

plot(r_Ls, lambdas_bar, 'Color', 'blue', 'Marker','x');
plot(r_Ls, lambda_l * ones(length(sat_bounds), 1), 'Color', 'green', 'LineStyle', '--');


%% Viuslaize Successive PRS

horizon = 10; %(recommended)

r_lambda = [];
r_lambda_l = [];
r_lambda_bar = [];

for i = 1:horizon

    r_lambda = [r_lambda; (1 - lambda^i)/(violation_prob * (1 - lambda)) * trace(P * W)];
    r_lambda_l = [r_lambda_l; (1 - lambda_l^i)/(violation_prob * (1 - lambda_l)) * trace(P * W)];
    r_lambda_bar = [r_lambda_bar; (1 - lambda_bar^i)/(violation_prob * (1 - lambda_bar)) * trace(P * W)];

end

figure(4)
hold on;
axis equal;
ylim([-40, 40])
xlim([-40, 50*(horizon)])
xticks(0:50:(horizon-1)*50)

for i = 1:horizon
    ee = sdpvar(n,1);

    h = plot(ee'*P*ee <= r_lambda(i, 1),ee,[],[]);
    
    % Apply artificial shift along x-axis
    ellipse_points = h{1}(1, :) + 1i * h{1}(2, :);
    ellipse_points = ellipse_points + (i-1) * 50;
    fill(real(ellipse_points), imag(ellipse_points), 'b', 'FaceAlpha', 0, 'EdgeColor', 'k', 'LineWidth', 2);

    h = plot(ee'*P*ee <= r_lambda_l(i, 1),ee,[],[]);
    ellipse_points = h{1}(1, :) + 1i * h{1}(2, :);
    ellipse_points = ellipse_points + (i-1) * 50;
    fill(real(ellipse_points), imag(ellipse_points), 'b', 'FaceAlpha', 0, 'EdgeColor', 'g', 'LineWidth', 2);

    h = plot(ee'*P*ee <= r_lambda_bar(i, 1),ee,[],[]);
    ellipse_points = h{1}(1, :) + 1i * h{1}(2, :);
    ellipse_points = ellipse_points + (i-1) * 50;
    fill(real(ellipse_points), imag(ellipse_points), 'b', 'FaceAlpha', 0, 'EdgeColor', 'b', 'LineWidth', 2);

    plot(X((i)*samples+1:(i+1)*samples, 1)+(i-1) * 50, X((i)*samples+1:(i+1)*samples, 2), '.r')

end

xt = get(gca, 'XTick');
set(gca, 'XTick', xt, 'XTickLabel', xt/50 + 1)