function P = pathConfig()
% pathConfig - Single entry for data/output paths; when moving the project, edit mainly P.root* and P.rootGrainIndexLegacy.

    here = fileparts(mfilename('fullpath'));
    projectRoot = fullfile(fileparts(here),'Revision');

    P.codeDir = here;
    P.projectRoot = projectRoot;

    % ----- Editable: case output roots (here under projectRoot; adjust paths as needed) -----
    P.rootNi100 = fullfile(projectRoot, 'Ni100');
    P.rootW100 = fullfile(projectRoot, 'W100_1.01');
    P.rootW100_100 = fullfile(projectRoot, 'W100_1.00');
    P.rootCr100 = fullfile(projectRoot, 'Cr100');
    P.rootGrainIndexLegacy = fullfile(projectRoot, 'GrainIndex');

    % ----- Outputs next to this code folder -----
    P.energyMat = fullfile(here, 'energy.mat');
    P.welsPng = fullfile(here, 'wels.png');

    % ----- Path builders (pass e.g. P.rootNi100 as root) -----
    P.giCsv = @(root, k) fullfile(root, 'output', 'GrainIndeics', 'GrainIndex', sprintf('GI_%d.csv', k));
    P.giCrCsv = @(root, k) fullfile(root, 'output', 'GrainIndeics', 'GrainIndex', sprintf('GI_%d.csv', k));
    P.textureGiPng = @(root, k) fullfile(root, 'output', 'GrainIndeics', 'TexturePngs', sprintf('GI_%d.png', k));
    P.grainTextureVideo = @(root) fullfile(root, 'output', 'GrainIndeics', 'TexturePngs', 'video.mp4');
    P.f0Fe = @(root, k) fullfile(root, 'output','GrainIndeics', 'F0', sprintf('FE_%d.csv', k));
    P.f1Ge = @(root, k) fullfile(root, 'output','GrainIndeics', 'F1', sprintf('GE_%d.csv', k));
    P.f2We = @(root, k) fullfile(root, 'output','GrainIndeics', 'F2', sprintf('WE_%d.csv', k));
    P.legacyGi = @(k) fullfile(P.rootGrainIndexLegacy, sprintf('GI_%d.csv', k));
end
