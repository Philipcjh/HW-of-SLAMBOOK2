clear all;
clc;
% 事实，1代表True，0代表False
% 这里没有数据，使用随机0，1的数组代替
Fact = round(rand(100, 1));

% 算法给出当前回环的打分
% 这里没有数据，使用随机0~1之间的实数数组代替
Score = rand(100, 1);

% 设置多个阈值，作为判断回环的置信度
threshold = linspace(0.9, 0.3, 10);

% 初始化，开辟内存空间
P = zeros(length(threshold), 1);
R = zeros(length(threshold), 1);

for i = 1 : length(threshold)
    [P(i), R(i)] = PRCalculate(Fact, Score, threshold(i));
end

plot(R, P, '-r');
xlabel('Recall');
ylabel('Precision');
legend('Random');
grid on;

%% 输入事实、分数和置信度阈值，返回准确率和召回率
function [P,R] = PRCalculate(Fact, Score, threshold)
TP = 0;
FP = 0;
FN = 0;
for i = 1 : length(Score)
    % 大于或等于阈值时，算法认为该回环为P
    if Score(i) >= threshold
        if Fact(i) == 1
            TP = TP + 1;
        else 
            FP = FP + 1;
        end
    % 小于阈值时，我们认为该回环为N
    elseif Score(i) < threshold && Fact(i) == 1
            FN = FN + 1;
    end
end        
P = TP / (TP + FP);
R = TP / (TP + FN);
end
