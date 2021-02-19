load imagenet
I=imread("images\454.png");

%(Optional: Noise example)
%I3 = insertShape(I,'line',[23 25 25 25],'LineWidth',10); % for two
%I3 = insertShape(I,'line',[9 14 16 14],'LineWidth',1); %for one

 figure
 imshow(I)
 truesize([600 400]);
 title('Input image')
% print(gcf,'9_851.png','-dpng','-r400');

n_predict = predict(net,I);
n_classify = classify(net,I);
 
prediction_pro = max(n_predict);
prediction_index=find(prediction_pro==n_predict);
% prediction_pro = n_predict(10); % To test a contrastive case
% prediction_index = 10;

L = superpixels(I, 9);
B = labeloverlay(I, L);
figure
imshow(B)
truesize([600 400]);
% print(gcf,'piximage1_851.png','-dpng','-r400');

num2=unique(L);
siz = size(num2,1);

[partition, prediction] = findsuperpixel(L, I, siz, net, prediction_index);
% prediction_pro=round(prediction_pro,10); % To test a contrastive case
% prediction=round(prediction,10);
importance=prediction_pro-prediction;

[imageq,imageq2] = saliencymap(L, I, importance, siz);

n2_predict = predict(net,imageq2);
n2_classify = classify(net,imageq2);

figure
imshow(imageq)
truesize([600 400]);
title('Importance')
% print(gcf,'9_851_n.png','-dpng','-r400');

function [d, prediction] = findsuperpixel(M, ima, siz, net, prediction_index)
prediction = zeros(1,siz)';

for l=1:siz   % find the superpixel
    k = M ~= l;       
    o = M*0;
    o(k) = 1;
    o = uint8(o);
    o = ima.*o;
    d=o;  
    
    predicted = predict(net,d);
    predicted = predicted(prediction_index);
    prediction(l)=predicted;
    
end

end

function [imageq,imageq2] = saliencymap(M, ima, importance, siz)

colorMap = [linspace(0,0.9,256)', linspace(0,0.25,256)', linspace(0,0.1,256)']; %red
colorMap2 = [linspace(0,1,256)', linspace(0,1,256)', linspace(0,1,256)']; %white

imageq=zeros(28,28);
imageq=ind2rgb(uint8(imageq),colorMap);

imageq2=zeros(28,28);
imageq2=uint8(imageq2);

for l=1:siz
    k = M == l;       
    o = M*0;
    o(k) = 1;
    o = uint8(o);
    o = ima.*o;
       
    if importance(l) >=0.01  && importance(l) <= 1       
        im = ind2rgb(o,colorMap); %red
        imageq=imageq+im;
        imageq2=imageq2+o;
    else
        im = ind2rgb(o,colorMap2); %white
        imageq=imageq+im;
    end
end
end
