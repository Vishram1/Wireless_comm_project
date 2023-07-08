function [convnetChannelEstimation] = Train(X,Y,l_rate)
%% Train network all the parameters that were used are used from the paper.
dataFRFChainSelection = X;
labelsRFChainSelection = Y;
sizeInputFRFChainSelection = size(dataFRFChainSelection);
sizeOutputFRFChainSelection = size(labelsRFChainSelection);
% val. for regression.
idx = randperm(size(dataFRFChainSelection,4),floor(.2*sizeInputFRFChainSelection(end)));
valDataRFChainSelection = dataFRFChainSelection(:,:,:,idx);
valLabelsFRFChainSelection = labelsRFChainSelection(idx,:);
dataFRFChainSelection(:,:,:,idx) = [];
labelsRFChainSelection(idx,:) = [];
% settings.
miniBatchSize = 32;
validationFrequency = 50*1;
layersFRFChainSelection = [imageInputLayer([sizeInputFRFChainSelection(1:3)],'Normalization', 'zerocenter');
    convolution2dLayer(3,2^8,'Padding','same');
%     batchNormalizationLayer
    convolution2dLayer(3,2^8,'Padding','same');
    fullyConnectedLayer(2^10);
    fullyConnectedLayer(2^10);
    fullyConnectedLayer(sizeOutputFRFChainSelection(2),'Name','fc_2')
    regressionLayer('Name','reg_out')
    ];
optsFRFSelection = trainingOptions('sgdm',...
    'Momentum', 0.9,...
    'InitialLearnRate',l_rate,...
    'MaxEpochs',500,...
    'MiniBatchSize',miniBatchSize,... 
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',.3,...
    'LearnRateDropPeriod',10,...
    'L2Regularization',0.0000001,....
    'ExecutionEnvironment', 'auto',...
    'ValidationData',{valDataRFChainSelection,valLabelsFRFChainSelection},...
    'ValidationFrequency',validationFrequency,...
    'ValidationPatience', 20000,...
    'Plots','none',...
    'Shuffle','every-epoch',...
    'OutputFcn',@(info)Callback(info,5));
convnetChannelEstimation = trainNetwork(dataFRFChainSelection, labelsRFChainSelection, layersFRFChainSelection, optsFRFSelection);