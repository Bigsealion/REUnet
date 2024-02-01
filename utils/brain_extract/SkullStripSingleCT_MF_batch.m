% extract brian, batch version
% please input:
%    source_dir, out_dir, start_index, end_index

tic
% print input parameters -------------------------------------------
fprintf('StripSkullCT_code_dir: %s\n', StripSkullCT_code_dir)
fprintf('source_dir: %s\n', source_dir)
fprintf('out_dir: %s\n', out_dir)
fprintf('start_index: %d\n', start_index)
fprintf('end_index: %d\n', end_index)

% add StripSkullCT-master ------------------------------------------
addpath(genpath(StripSkullCT_code_dir));

% mkdir out dir -----------------------------------------------
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
    fprintf('mkdir Out Dir: %s\n', out_dir)
end

% get subj list ----------------------------------------------
file_info_stc = dir(fullfile(source_dir,'*.nii.gz'));
file_name_cell = {file_info_stc.name}';
file_path_cell = fullfile({file_info_stc.folder}', {file_info_stc.name}');

error_list = {};
error_num = 0;

% check end index
if end_index > length(file_name_cell) || end_index <= 0
    fprintf('input end index is %d, but file number is %d\n', end_index, length(file_name_cell))
    fprintf('set end_index to file number\n')
    end_index = length(file_name_cell); 
end

if start_index <= 0
    fprintf('input staet index <= 0, using all %d case in source dir!\n', length(file_name_cell))
    start_index = 1;
    end_index = length(file_name_cell);
end

% run batch ---------------------------------------------------------
for i = start_index: end_index
    pathNCCTImage = [file_path_cell{i}];
    PathNCCT_Brain = [fullfile(out_dir, file_name_cell{i})];
    
    fprintf('-----> %d/%d %s %.2fs\n', i, length(file_name_cell), file_name_cell{i}, toc)
    disp(['    Strip skull of patient ' pathNCCTImage]);
    
    % load the subject image
    ImgSubj_nii = load_untouch_nii(pathNCCTImage);
    ImgSubj_hdr = ImgSubj_nii.hdr;
    ImgSubj = ImgSubj_nii.img;
    %ImgSubj = double(ImgSubj);
    
    % skull stripping
    NCCT_Thr = 100; % for NCCT images
    CTA_Thr = 400; % for CTA images
    
    % remove the effect of the intercept
    ImgSubj = ImgSubj + ImgSubj_hdr.dime.scl_inter;
    
    try
        [brain] = SkullStripping(double(ImgSubj),NCCT_Thr);
    catch
        error_num = error_num + 1;
        error_list{error_num} = file_name_cell{i};
        fprintf('\tError: %s\n', file_name_cell{i})
        continue
    end
    
    % remove the effect of the intercept
    brain = brain - ImgSubj_hdr.dime.scl_inter;
    
    % save image
    
    Output_nii.hdr = ImgSubj_hdr;
    Output_nii.img = int16(brain);
    save_nii(Output_nii, PathNCCT_Brain);
    
    disp(['    ', pathNCCTImage '----skull tripping finished']);
end

% print error 
if ~isempty(error_list)
    fprintf('-----> Error subject:')
    for i = 1: length(error_list)
        fprintf('\t%s\n', error_list{i})
    end
end

