% run_tracker.m

close all; clear;
%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'E:\DataSets\CFTracker\People';

% CN参数
params.CN.padding = 1.0; % 目标周围的额外区域
params.CN.output_sigma_factor = 1/16; % 空间带宽（与目标成比例）
params.CN.sigma = 0.2; % 高斯核带宽
params.CN.lambda = 1e-2; % 正则化（文中用“λ”表示）
params.CN.learning_rate = 0.075; % 外观模型更新方案的学习率（文中用“γ”表示）
params.CN.compression_learning_rate = 0.15; %自适应降维的学习率（文中用“μ”表示）
params.CN.non_compressed_features = {'gray'}; %未压缩的特征，一个带有字符串的元包（可能的选项：gray，cn）
params.CN.compressed_features = {'cn'}; % 压缩的特征，一个带有字符串的元包（可能的选项：gray，cn）
params.CN.num_compressed_dim = 2; %压缩特征的维数

% fDSST参数
% 平移滤波器参数
params.fDSST.padding = 2.0; % 目标周围的额外区域
params.fDSST.output_sigma_factor = 1/16; % 所需平移滤波器输出的标准差
params.fDSST.lambda = 1e-2; % 正则化权重（在论文中用“λ”表示）
params.fDSST.interp_factor = 0.025; % 跟踪模型学习率（文中用“η”表示）
params.fDSST.num_compressed_dim = 18; % 压缩特征的维数
params.fDSST.refinement_iterations = 1; % 用于在帧中细化结果位置的迭代次数
params.fDSST.translation_model_max_area = inf; % 平移模型的最大面积
params.fDSST.interpolate_response = 1; % 平移响应分数的差值方法
params.fDSST.resize_factor = 1; % 初始重置大小
% 尺度滤波器参数
params.fDSST.scale_sigma_factor = 1/16; % 所需尺度滤波器输出的标准差
params.fDSST.number_of_scales = 17; % 尺度数
params.fDSST.number_of_interp_scales = 33; % 差值后尺度数
params.fDSST.scale_model_factor = 1.0; % 尺度样本的相对大小
params.fDSST.scale_step = 1.02; % 比例增量系数（在文件中用"a"表示）
params.fDSST.scale_model_max_area = 512; % 尺度样本的最大大小
params.fDSST.s_num_compressed_dim = 'MAX'; % 压缩尺度特征维度数

params.visualization = 0;
params.draw_apce = 0;

% 加载视频序列
dirs = dir(base_path);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos) | ...
            strcmp('anno', videos) | ~[dirs.isdir]) = [];

videos(strcmpi('Jogging', videos)) = [];
videos(end + 1:end + 2) = {'Jogging.1', 'Jogging.2'};

all_DP = zeros(numel(videos), 1);
all_OP = zeros(numel(videos), 1);
all_CLE = zeros(numel(videos), 1);
all_fps = zeros(numel(videos),1);

for k = 1:numel(videos)
    [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, videos{k});
    
    params.init_pos = floor(pos); %初始位置[y,x]，左上角
    params.wsize = floor(target_sz); % 目标大小[h,w]
    params.img_files = img_files; %视频序列的集合
    params.video_path = video_path; %视频序列根路径
    
    results = tracker(params);
    % positions = results.positions;
    boxes = results.boxes;
    all_fps(k) = results.fps;
    
    % calculate precisions
    [all_DP(k), all_OP(k), all_CLE(k)] = compute_performance_measures(boxes, ground_truth);
    fprintf('%12s - DP (20px):% 1.3f, OP:%1.3f, CLE (pixcel):%.5g, fFPS:% 4.2f\n', videos{k}, all_DP(k), all_OP(k), all_CLE(k), all_fps(k));
end

fprintf('Average - DP (20px):% 1.3f, OP:%1.3f, CLE (pixcel):%.5g, fFPS:% 4.2f\n', mean(all_DP), mean(all_OP), mean(all_CLE), mean(all_fps));