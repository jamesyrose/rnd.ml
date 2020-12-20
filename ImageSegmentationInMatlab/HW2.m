% 
% James Rose
% ECE 271a 
% Problem Set 2
% 
load('TrainingSamplesDCT_8.mat')

%%%%%%%%%%% PART A %%%%%%%%%%%%%%

indexs = zeros(1, size(TrainsampleDCT_FG, 1) ...
                  +  size(TrainsampleDCT_BG, 1) ); 
n_cheetah = size(TrainsampleDCT_FG, 1);
n_grass = size(TrainsampleDCT_BG, 1); 
values = cat(1, ones(n_cheetah, 1),  zeros(n_grass, 1));

pi_vals = 0:.0001:1;
var = 1;  %% Assuming Variance for Pi

pi_results = zeros(1, length(pi_vals));
for i=1:length(pi_vals)
    % Gaussian
    pi_results(i) = exp(-.5 * sum((values - pi_vals(i)).^2)/var);
end
% Pi Found
[pi_max, pi_idx] =  max(pi_results);
% Priors
pi_cheetah = pi_vals(pi_idx);
pi_grass = 1 - pi_cheetah;

% True variance
variance =  sum((values - pi_cheetah).^2) / length(values);

msg = join(["The results are the same as in the first homework. This"...
    " makes sense because the distribution can be seen as binomial " ...
    "(cheetah or grass). This means the that hessian should benegative" ... 
    "definite along the diagonal and therefore concave.  \n"... 
    "We can also see this if we interpret the MLE as P(Data | theta) where" ...
    "theta is the probability of say cheetah. Finding argmax(theta)"...
    "P(data|theta) where the distribution is binmoial, we would find out "...
    "that theta would be equal to n(cheetah) , or the initial assumption"...
    " made in hw1. In addition, in problem 2 we shoed that the mean " ...
    "estimator was equal to n * pi_i / n or just the same priors in HW1\n" 
    ], " ");
disp("Part A");
fprintf("Pi_Cheetah: %.4f \n", pi_cheetah);
fprintf("Pi_Grass: %.4f \n", pi_grass);
fprintf("Variance: %.4f \nSTD: %.4f \n", variance, sqrt(variance));
disp(msg)


% %%%%%%%%%%% PART B  %%%%%%%%%%%%%%

fg_indexs = zeros(1, size(TrainsampleDCT_FG, 1)); % P(x|cheetah)
bg_indexs =  zeros(1, size(TrainsampleDCT_BG, 1)); % P(x|grass)


n_grass = size(TrainsampleDCT_BG, 1);
n_cheetah = size(TrainsampleDCT_FG, 1);
pi_vals = 0:.001:2;
def_var = 1;
grass_pi_res = zeros(64,1);
cheetah_pi_res= zeros(64,1);
grass_var_res = zeros(64,1);
cheetah_var_res= zeros(64,1);

figure(1)
for i = 1:64
    grass_vals = zeros(length(TrainsampleDCT_FG), 1);

    for j=1:length(TrainsampleDCT_FG)
        grass_vals(j) = TrainsampleDCT_FG(j, i);
    end
    cheetah_vals = zeros(length(TrainsampleDCT_BG), 1);
    for j=1:length(TrainsampleDCT_BG)
        cheetah_vals(j) = TrainsampleDCT_BG(j, i);
    end
    
    n_grass =  length(grass_vals);
    n_cheetah = length(cheetah_vals);
    
    grass_res = zeros(n_grass, 1);
    cheetah_res = zeros(n_cheetah, 1);
    
    for j=1:length(pi_vals)
        grass_res(j) = exp(-.5 * sum((grass_vals - pi_vals(j)).^2)/def_var);
    end
    for j=1:length(pi_vals)
        cheetah_res(j) = exp(-.5 * sum((cheetah_vals - pi_vals(j)).^2)/def_var);
    end
    [gm, gi] = max(grass_res);
    [cm, ci] = max(cheetah_res);
    
    grass_pi = pi_vals(gi);
    cheetah_pi = pi_vals(ci);

    grass_var = sum((grass_vals - grass_pi).^2) / (n_grass);
    cheetah_var = sum((cheetah_vals - cheetah_pi).^2) / (n_cheetah);
    
    grass_pi_res(i) = grass_pi;
    grass_var_res(i) = grass_var;
    cheetah_pi_res(i) = cheetah_pi;
    cheetah_var_res(i) = cheetah_var;
    
    
    allvals = cat(1, grass_vals, cheetah_vals);
    x_vals = min(allvals):.0001:max(allvals);
    grass_dist = gaussianPDF(x_vals, grass_pi, sqrt(grass_var));
    cheetah_dist = gaussianPDF(x_vals, cheetah_pi, sqrt(cheetah_var));

    subplot(8,8,i)
    l1 = plot(x_vals, grass_dist, "Color", "g");
    hold on
    l2 = plot(x_vals, cheetah_dist, "Color", "r", "LineStyle", "--");
    hold on
    title(sprintf("Index: %d", i));
end
hold off
hl = legend([l1,l2], {"Grass", "Cheetah"});
set(hl, "Position", [.1, .1, .2, .2]);
sgtitle("Gaussian Dist of each index");


% Best 8 (Visual)
disp("Best 8 Based on visual")
disp("Index: 1, 12, 15, 17, 23, 24, 25, 32")
disp("These are the best features because we can see the distributions for each feature given grass or cheetah are very different. That is the distirbution overlaps the least. Additionally, low variance is ideal because it means the distribution is more certain.")
figure(2)
features = [1, 12, 15,17,23,24,25, 32];
for idx=1:length(features)
    i = features(idx);
    % Getting values again 
        grass_vals = zeros(length(TrainsampleDCT_FG), 1);

    for j=1:length(TrainsampleDCT_FG)
        grass_vals(j) = TrainsampleDCT_FG(j, i);
    end
    cheetah_vals = zeros(length(TrainsampleDCT_BG), 1);
    for j=1:length(TrainsampleDCT_BG)
        cheetah_vals(j) = TrainsampleDCT_BG(j, i);
    end
    % Plottings
    grass_pi = grass_pi_res(i);
    grass_var = grass_var_res(i);
    cheetah_pi = cheetah_pi_res(i);
    cheetah_var = cheetah_var_res(i);
    
    allvals = cat(1, grass_vals, cheetah_vals);
    x_vals = min(allvals):.0001:max(allvals);
    grass_dist = gaussianPDF(x_vals, grass_pi, sqrt(grass_var));
    cheetah_dist = gaussianPDF(x_vals, cheetah_pi, sqrt(cheetah_var));
    subplot(2,4,idx)
    l1 = plot(x_vals, grass_dist, "Color", "g");
    hold on
    l2 = plot(x_vals, cheetah_dist, "Color", "r", "LineStyle", "--");
    hold on
    title(sprintf("Index: %d", i));
end
hold off
sgtitle("Best 8 Features")

% Worst 8 (Visual)
disp("Worst 8 Visual");
disp("Index: 2,3,4,5,6,58,59,64")
disp("The worst features would be those that have the most overlap because the greater the overlap means the more similar the features between cheeath and grass.");

figure(3)
features = [2,3,4,5,6,58,59,64];
for idx=1:length(features)
    i = features(idx);
    % Getting values again 
        grass_vals = zeros(length(TrainsampleDCT_FG), 1);

    for j=1:length(TrainsampleDCT_FG)
        grass_vals(j) = TrainsampleDCT_FG(j, i);
    end
    cheetah_vals = zeros(length(TrainsampleDCT_BG), 1);
    for j=1:length(TrainsampleDCT_BG)
        cheetah_vals(j) = TrainsampleDCT_BG(j, i);
    end
    % Plottings
    grass_pi = grass_pi_res(i);
    grass_var = grass_var_res(i);
    cheetah_pi = cheetah_pi_res(i);
    cheetah_var = cheetah_var_res(i);
    
    allvals = cat(1, grass_vals, cheetah_vals);
    x_vals = min(allvals):.0001:max(allvals);
    grass_dist = gaussianPDF(x_vals, grass_pi, sqrt(grass_var));
    cheetah_dist = gaussianPDF(x_vals, cheetah_pi, sqrt(cheetah_var));
    subplot(2,4,idx)
    l1 = plot(x_vals, grass_dist, "Color", "g");
    hold on
    l2 = plot(x_vals, cheetah_dist, "Color", "r", "LineStyle", "--");
    hold on
    title(sprintf("Index: %d", i));
end
hold off
sgtitle("Worst 8 Features")

%%%%%%%%%%%%%%%%% Part C %%%%%%%%%%%%%%%%%

img = im2double(imread("cheetah.bmp"));
mask_all = zeros(size(img,1), size(img, 2));
mask_feature = zeros(size(img,1 ), size(img,2));
for i=1:(size(img, 1) - 8)
    for j=1:(size(img, 2) - 8)
       block = img(i:i+8, j:j+8);
       blockDCT = dct2(block);
       blockFlat = flatten(blockDCT);
       
       grass_probs = zeros(64,1);
       cheetah_probs = zeros(64, 1);
       
       features = [1, 12, 15,17,23,24,25, 32]; 
       grass_probs_features = zeros(64, 1);
       cheetah_probs_features = zeros(64, 1);
       for k = 1:64
           val = blockFlat(k);
           grass_pi = grass_pi_res(k);
           grass_var = grass_var_res(k);
           cheetah_pi = cheetah_pi_res(k);
           cheetah_var = cheetah_var_res(k);
           
           
           grass_fun = @(x) 1/(sqrt(grass_var) * sqrt(2 *pi)) * ...
               exp(-.5 / grass_var * (x - grass_pi) .^2);
           cheetah_fun = @(x) 1/(sqrt(cheetah_var) * sqrt(2 *pi)) * ...
               exp(-.5 / cheetah_var * (x - cheetah_pi) .^2 );
            
           grass_buff = grass_fun(val);
           cheetah_buff = cheetah_fun(val);
           
           grass_prob = grass_buff/ (grass_buff + cheetah_buff) * pi_grass;
           cheetah_prob = cheetah_buff / (grass_buff + cheetah_buff) * pi_cheetah;
      
           
           grass_probs(k) = grass_prob * pi_grass;
           cheetah_probs(k) = cheetah_prob * pi_cheetah;
                      

       end
       for k = 1:features
           val = blockFlat(k);
           grass_pi = grass_pi_res(k);
           grass_var = grass_var_res(k);
           cheetah_pi = cheetah_pi_res(k);
           cheetah_var = cheetah_var_res(k);
           
           
           grass_fun = @(x) 1/(sqrt(grass_var) * sqrt(2 *pi)) * ...
               exp(-.5 / grass_var * (x - grass_pi) .^2);
           cheetah_fun = @(x) 1/(sqrt(cheetah_var) * sqrt(2 *pi)) * ...
               exp(-.5 / cheetah_var * (x - cheetah_pi) .^2 );
            
           grass_buff = grass_fun(val);
           cheetah_buff = cheetah_fun(val);
           
           grass_prob = grass_buff/ (grass_buff + cheetah_buff) * pi_grass;
           cheetah_prob = cheetah_buff / (grass_buff + cheetah_buff) * pi_cheetah;
      
           
           grass_probs_features(k) = grass_prob * pi_grass;
           cheetah_probs_features(k) = cheetah_prob * pi_cheetah;
                      

       end
       
       grass_max = max(grass_probs);
       cheetah_max = max(cheetah_probs);
       if grass_max > cheetah_max
           mask_all(i, j) = 0;
       else
           mask_all(i, j) = 1;
       end
       
       grass_max = max(grass_probs_features);
       cheetah_max = max(cheetah_probs_features);
       if grass_max > cheetah_max
           mask_feature(i, j) = 1;
       else
           mask_feature(i, j) = 0;
       end
    end
end

mask = im2double(imread("cheetah_mask.bmp"));

mask_all_error = sum(sum(abs(mask_all - mask))) / (255*270);
mask_feature_error = sum(sum(abs(mask_feature - mask))) / (255*270);


figure(4)
title("Mask with no selected features")
imagesc(img); 
hold on
imagesc(mask_all, 'AlphaData', .5);
colormap(gray(255));
hold off

fprintf("Error: %.5f", mask_all_error);
disp("Since were are predicting cheetah vs grass using argmax, using all 64 features intruduces a lot of noise in the prediction. Because of this and the probability of it being grass, we end up with a very high likely hood that the argmax will yield a prediction of grass.")

figure(5)
title("Mask with best 8 Features")
imagesc(img)
hold on
imagesc(mask_feature, 'AlphaData', .5);
colormap(gray(255));
hold off

fprintf("Error: %.5f", mask_feature_error); 
disp("The seleceted features yield better prediction of whether or not spach is cheetah or grass. So by narrowing the selection to just those 8 features, we remove the excess noise from features that do not predict well. Thus, taking argmax of the each features prediction will make it fairly accurate. Likely it would be better to use all 8 features and weight them  in accordance to the respective tested accuracy. that way we get the summation of the P(Cheetah|feature) * P(feature accuractely predicted the result)");
















