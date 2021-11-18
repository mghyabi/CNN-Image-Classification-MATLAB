%% i)
Arch=cell(numel(net.Layers),5);
for i=[2 6 10]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='Convolution2D';
    Arch{i,3}=l.NumFilters;
    Arch{i,4}=num2str(l.FilterSize);
    Arch{i,5}=numel(l.Weights)+numel(l.Bias);
end
for i=[3 7 11]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='BatchNOrmalization';
end
for i=[4 8 12]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='ReLU';
end
for i=[5 9]
    l=net.Layers(i);
    Arch{i,1}=l.Name;
    Arch{i,2}='MaxPooling';
end
l=net.Layers(14);
Arch{14,1}=l.Name;
Arch{14,2}='Softmax';
l=net.Layers(15);
Arch{15,1}=l.Name;
Arch{15,2}='ClassificationOutput';
l=net.Layers(1);
Arch{1,1}=l.Name;
Arch{1,2}='ImageInput';
l=net.Layers(13);
Arch{13,1}=l.Name;
Arch{13,2}='FullyConnected';
Arch{13,5}=numel(l.Weights)+numel(l.Bias);
table(Arch)
%% ii)
for i=[10]
    layer=net.Layers(i);
    W=layer.Weights;
    a=ceil(sqrt(size(W,3)*size(W,4)));
    figure
    l=1;
    for j=1:size(W,3)
        for k=1:size(W,4)
            subplot(a,a,l)
            imagesc(W(:,:,j,k))
            axis off
            colormap gray
            l=l+1;
        end
    end
    suptitle(['Layer Number ' num2str(i)])
end

%% iii)
im=imgDataTest(:,:,1,1);
for i=[2 6 10]
    act=activations(net,im,Arch{i,1});
    a=ceil(sqrt(size(act,3)));
    figure
    for j=1:size(act,3)
        subplot(a,a,j)
        imagesc(act(:,:,j))
        axis off
        colormap gray
    end
    suptitle(['Results from Layer Number ' num2str(i)])
end

%% iv)
for i=[2 6 10]
   maxact=deepDreamImage(net,i,1:Arch{i,3},'PyramidLevels',1);
   a=ceil(sqrt(size(maxact,4)));
   figure
   for j=1:size(maxact,4)
       subplot(a,a,j)
       imagesc(maxact(:,:,1,j))
       axis off
       colormap gray
   end
   suptitle(['Results from Layer Number ' num2str(i)])
end