%% i:Preparing data
% size of input images
Size=128;
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
        im=imresize(im,[128 128]);
        imgDataTest=cat(4,imgDataTest,im);
    end
end
labelsTrain=categorical(labelsTrain);
labelsTest=categorical(labelsTest);

%% ii: MNIST implimentation
layers = [
    imageInputLayer([128 128 3])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(101)
    softmaxLayer
    classificationLayer];
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', 8192,...
    'Plots', 'training-progress');
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);

predLabelsTest = net.classify(imgDataTest);
testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)

%%
[x,y]=meshgrid(unique(labelsTest),unique(labelsTest));
Pred=repmat(reshape(predLabelsTest,1,1,[]),numel(unique(labelsTest)),numel(unique(labelsTest)));
Actual=repmat(reshape(labelsTest,1,1,[]),numel(unique(labelsTest)),numel(unique(labelsTest)));
Confusion_Matrix=sum((((Actual==y)+(Pred==x))==2),3);
Confusion_Matrix=Confusion_Matrix./repmat(sum(Confusion_Matrix,2),1,size(Confusion_Matrix,2));
Confusion_Matrix=Confusion_Matrix/max(Confusion_Matrix(:));
figure,
imshow(Confusion_Matrix)