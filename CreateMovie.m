clear all; clc; close all;

P = pathConfig;
zTolerance = 1e-8;
zValueToKeep = 1;
x_col = 3; y_col = 4; z_col = 5; gi_col = 2;

rootNi = P.rootNi100;

for Num = 0:95
    filePath_GI = P.giCsv(rootNi, Num);
    filePath_outputPng = P.textureGiPng(rootNi, Num);
    [time, maxGI, GI] = readCSVfile(filePath_GI);
    fig_GI = plotOrientationMap(imresize(GI,2,"nearest"),'ShowColorbar', false,'Verbose', false);
    exportgraphics(fig_GI, filePath_outputPng,'Resolution',600,'BackgroundColor','white')
    close all;
end


videoFile = P.grainTextureVideo(rootNi);
videoObj = VideoWriter(videoFile, 'MPEG-4');  % or 'Uncompressed AVI', etc.
videoObj.FrameRate = 10;  % frames per second (tune as needed)
open(videoObj);

for Num = 0:95
    filePath_outputPng = P.textureGiPng(rootNi, Num);
    frame = imread(filePath_outputPng);
    writeVideo(videoObj, im2uint8(frame));
    fprintf('Wrote frame %d\n', Num);
end

% Close and finalize video file
close(videoObj);

disp('Video assembly finished.');
