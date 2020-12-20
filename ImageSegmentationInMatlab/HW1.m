% 
% James Rose
% ECE 271a 
% Problem Set 1
% 
load('TrainingSamplesDCT_8.mat')

%%%%%%%%%%%%%%%%%% Part A %%%%%%%%%%%%%%%%%%

% Since the trainsample BG and FG are repsenetitive of index's it found 
% to be background or foregound, it would be inuitive to say the ratio of
% the two represent the ratio of cheetah and grass to the picture.
totalSize = size(TrainsampleDCT_BG, 1) + size(TrainsampleDCT_FG, 1);
P_cheetah = size(TrainsampleDCT_FG, 1) / totalSize;   %  P(Cheetah)
P_grass = size(TrainsampleDCT_BG, 1) / totalSize;     %  P(Grass)
fprintf("P(Cheetah): %.6f \nP(Grass): %.6f\n", P_cheetah, P_grass);

%%%%%%%%%%%%%%%%%% Part B %%%%%%%%%%%%%%%%%%


fg_indexs = zeros(1, size(TrainsampleDCT_FG, 1)); % P(x|cheetah)
bg_indexs =  zeros(1, size(TrainsampleDCT_BG, 1)); % P(x|grass)

for i = 1:size(TrainsampleDCT_FG)
    arr = TrainsampleDCT_FG(i, :);
    arr = abs(arr);
    secondMax = max(arr(arr<max(arr)));
    idx = find(arr==secondMax);
    fg_indexs(i) = idx;
end
for i = 1:size(TrainsampleDCT_BG)
    arr = TrainsampleDCT_BG(i, :);
    arr = abs(arr);
    secondMax = max(arr(arr<max(arr)));
    idx = find(arr==secondMax);
    bg_indexs(i) = idx;
end

figure("name", "Index");
grassH = histogram(bg_indexs,'Normalization','probability', ...
                    'BinEdges', linspace(1, 65, 65), ...
                    'facealpha',.5,'edgecolor','none');
hold on; 
cheetahH = histogram(fg_indexs,'Normalization','probability', ...
                    'BinEdges', linspace(1, 65, 65), ...
                    'facealpha',.5,'edgecolor','none'); 
legend("Grass", "Cheetah", 'location', 'northeast');
title("P(X|Y)");
xlabel("Index"); 
ylabel("Count");
legend boxoff; box off ; axis tight;


%%%%%%%%%%%%%%%%%% Part C %%%%%%%%%%%%%%%%%%
dctZigZagMat  = [];

zigzag = readmatrix("Zig-Zag Pattern.txt");
img = im2double(imread("cheetah.bmp"));
img = normalize(img);
counter = 1;
row = 1; col = 1;
for i = 1:8:size(img, 1)
    for j = 1:8:size(img, 2) 
        if i+7 < size(img, 1) && j+7 < size(img, 2)
            block = img(i:i+7, j:j+7);
            blockDCT  = dct2(block);
            buff = flatten(blockDCT);
            dctZigZagMat(end+1 , :) = abs(buff);
        end
    end
end


index =  zeros(1, size(dctZigZagMat, 1)); 
for i = 1:size(dctZigZagMat)
    arr = dctZigZagMat(i, :);
    arr = abs(arr);
    secondMax = max(arr(arr<max(arr)));
    idx = find(arr==secondMax);
    index(i) = idx;
end

% Looking for P(y|x) = P(x|y) p(y) / p(x)
A = [];
for i = 1:size(index, 2)
    idx = index(1, i);
    P_CheetahX = cheetahH.Values(1, idx) * P_cheetah /  ...  % Bayes
        (cheetahH.Values(idx) * P_CheetahX + ... % Marginilization
        (1- cheetahH.Values(idx)) * (1-P_CheetahX));
    P_GrassX = grassH.Values(1, idx) * P_grass /   ... % Bayes
        (grassH.Values(idx) * P_GrassX + ... % Marginilization
        (1- grassH.Values(idx)) * (1-P_GrassX));
    if P_CheetahX > P_GrassX
        A(end + 1, :) = 1;
    else
        A(end + 1, :)  = 0;
    end
end

% Based on one feature (which likely isnt the best). It makes sense that it
% would not provide a discernable shape. However, it does show that it
% locates the cheeta towards the center of the map. Furthermore, it does
% better in the areas which are 'more' cheetah (rather than edges or the
% tail)
figure 
title("ColorMap of Predictions");
imagesc(rot90(fliplr(reshape(A, [33, 31]))));
colormap(gray(255)); 
% You can make out certain features, but by no means could anyone identify
% this as a cheetah in a grass prarie 

%%%%%%%%%%%%%%%%%% Part D %%%%%%%%%%%%%%%%%%
img = im2double(imread("cheetah_mask.bmp"));
actMask = [];
% Use the exact same method to get the act mask in the same shape. avoid
% error in structuring 
for i = 1:8:size(img, 1)
    for j = 1:8:size(img, 2) 
        if i+7 < size(img, 1) && j+7 < size(img, 2)
            block = flatten(img(i:i+7, j:j+7));
            if sum(block > 32) % Half the vals are cheetah
                isCheetah = 1;
            else 
                isCheetah = 0; 
            end
            actMask(end+1, 1) = isCheetah;
        end
    end
end 
wrong = 0;
total = 0;
for i = 1:size(actMask, 1)
    act = actMask(i);
    algo = A(i);
    if ~act==algo
        wrong = wrong + 1; 
    end
    total = total + 1; 
end
% Amount wrong
error = wrong / total;
fprintf("Error Rate: %.5f\n", error);
% Error rate is roughly 8.6%. This May not seem that high but it is pretty
% high. Since this is only a binary classification, where one option occurs
% roughly 80% of the time, it would mean if we just guess grass everytime,
% we would have  an 80% accuracy. However with this we have slightly better
% accuracy of 91.4%, which is not bad given how simplistic this model is
% and how it is based on a single feature. 






