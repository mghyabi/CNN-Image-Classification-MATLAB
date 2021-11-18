%% i)
net=vgg16;

%% ii)
Arch=cell(numel(net.Layers),5);
for i=[2 4 7 9 12 14 16 19 21 23 26 28 30]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='Convolution2D';
    Arch{i,3}=l.NumFilters;
    Arch{i,4}=l.FilterSize;
    Arch{i,5}=numel(l.Weights)+numel(l.Bias);
end
for i=[3 5 8 10 13 15 17 20 22 24 27 29 31 34 37]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='ReLU';
end
for i=[6 11 18 25 32]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='MaxPooling';
end
for i=[33 36 39]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='FullyConnected';
    Arch{i,5}=numel(l.Weights)+numel(l.Bias);
end
for i=[35 38]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='DropoutLayer';
end
l=net.Layers(40);
Arch{40,1}=l.Name;
Arch{40,2}='Softmax';
l=net.Layers(41);
Arch{41,1}=l.Name;
Arch{41,2}='ClassificationOutput';
l=net.Layers(1);
Arch{1,1}=l.Name;
Arch{1,2}='ImageInput';
table(Arch)

%% iii)
maxact=deepDreamImage(net,i,1:Arch{2,3},'PyramidLevels',1);
a=ceil(sqrt(size(maxact,4)));
figure
for j=1:size(maxact,4)
    subplot(a,a,j)
    imagesc(maxact(:,:,1,j))
    axis off
    colormap gray
end
suptitle(['Results from Layer Number ' num2str(2)])

%% iv)
Size=224;
categories=dir('101_ObjectCategories');
categories(1:2)=[];
Result=cell(numel(categories),2);
for k=1:numel(categories)
    Directory=categories(k).name;
    im=imread([cd '\101_ObjectCategories\' Directory '\image_0001.jpg']);
    %dealing with grayscale images
    if size(im,3)==1
        im=repmat(im,1,1,3);
    end
    im=imresize(im,[Size Size]);
    label = classify(net, im);
    Result{k,1}=Directory;
    Result{k,2}=char(label);
end

%%
Size=224;
categories=dir('101_ObjectCategories');
categories(1:2)=[];
imgDataTrain=[];
labelsTrain=[];
imgDataTest=[];
labelsTest=[];
for k=1:numel(categories)
    Directory=categories(k).name;
    Names1=dir(['101_ObjectCategories\' Directory '\']);
    Names1(1:2)=[];
    for i=1:floor(numel(Names1)*.9)
        im=imread([cd '\101_ObjectCategories\' Directory '\' Names1(i).name]);
        %dealing with grayscale images
        if size(im,3)==1
           im=repmat(im,1,1,3); 
        end
        im=imresize(im,[Size Size]);
        imgDataTrain=cat(4,imgDataTrain,im);
    end
    labelsTrain=[labelsTrain;k*ones(i,1)];
    labelsTest=[labelsTest;k*ones(numel(Names1)-i,1)];
    for j=i+1:numel(Names1)
        im=imread([cd '\101_ObjectCategories\' Directory '\' Names1(j).name]);
        %dealing with grayscale images
        if size(im,3)==1
           im=repmat(im,1,1,3); 
        end
        im=imresize(im,[Size Size]);
        imgDataTest=cat(4,imgDataTest,im);
    end
end
labelsTrain=categorical(labelsTrain);
labelsTest=categorical(labelsTest);
%% v)
% Input of the network is the same as part C
load('D0Data.mat')
net = vgg16;
% defining layers of CNN
for i=1:numel(net.Layers)-1
    layers(i,1)=net.Layers(i);
end
for i=[2 4 7 9 12 14 16 19 21 23 26 28 30]
    layers(i,1).BiasLearnRateFactor=0;
    layers(i,1).BiasL2Factor=0;
    layers(i,1).WeightLearnRateFactor=0;
    layers(i,1).WeightL2Factor=0;
end
layers(39,1)=fullyConnectedLayer(101,'Name','fc8','WeightL2Factor',0);
layers(41,1)=classificationLayer('Name','output');

options = trainingOptions('adam',...
    'MaxEpochs',15,...
    'Plots','training-progress');
clearvars net
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);

predLabelsTest = net.classify(imgDataTest);
testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)


