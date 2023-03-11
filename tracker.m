function results = tracker(params)

%% 参数加载与初始化
    %%%%%%%%%%%%%%%%%%%%%%%%% CN参数 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CN_padding = params.CN.padding; % 目标周围的额外区域
    CN_output_sigma_factor = params.CN.output_sigma_factor; % 空间带宽（与目标成比例）
    CN_sigma = params.CN.sigma; % 高斯核带宽
    CN_lambda = params.CN.lambda; % 正则化（文中用“λ”表示）
    CN_learning_rate = params.CN.learning_rate; % 外观模型更新方案的学习率（文中用“γ”表示）
    CN_compression_learning_rate = params.CN.compression_learning_rate; %自适应降维的学习率（文中用“μ”表示）
    CN_non_compressed_features = params.CN.non_compressed_features; %未压缩的特征，一个带有字符串的元包（可能的选项：gray，cn）
    CN_compressed_features = params.CN.compressed_features; % 压缩的特征，一个带有字符串的元包（可能的选项：gray，cn）
    CN_num_compressed_dim = params.CN.num_compressed_dim; %压缩特征的维数
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%% fDSST参数 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    padding = params.fDSST.padding;
    output_sigma_factor = params.fDSST.output_sigma_factor;
    lambda = params.fDSST.lambda;
    interp_factor = params.fDSST.interp_factor;
%     refinement_iterations = params.fDSST.refinement_iterations;
    translation_model_max_area = params.fDSST.translation_model_max_area;
    
    nScales = params.fDSST.number_of_scales;
    nScalesInterp = params.fDSST.number_of_interp_scales;
    scale_step = params.fDSST.scale_step;
    scale_sigma_factor = params.fDSST.scale_sigma_factor;
    scale_model_factor = params.fDSST.scale_model_factor;
    scale_model_max_area = params.fDSST.scale_model_max_area;
    interpolate_response = params.fDSST.interpolate_response;
    num_compressed_dim = params.fDSST.num_compressed_dim;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    video_path = params.video_path;
    img_files = params.img_files;
    pos = floor(params.init_pos);
    target_sz = floor(params.wsize);
    visualization = params.visualization; %可视化
    draw_apce = params.draw_apce;

    num_frames = numel(img_files); % 帧数
    
    %%%%%%%%%%%%%%%%%%%%%% CN模型初始化 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 加载标准化颜色名称矩阵
    temp = load('w2crs');
    w2c = temp.w2crs;
    use_dimensionality_reduction = ~isempty(CN_compressed_features);
    % 填充后的窗口大小
    CN_sz = floor(target_sz * (1 + CN_padding));
    % 期望输出（高斯形状），带宽与目标尺寸成比例
    CN_output_sigma = sqrt(prod(target_sz)) * CN_output_sigma_factor;
    CN_yf = single(fft2(gaussian_shaped_labels(CN_output_sigma, CN_sz)));
    % 存储预计算余弦窗口
    CN_cos_window = single(hann(CN_sz(1)) * hann(CN_sz(2))');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%%%%%%%%%%%%%% fDSST模型初始化 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    init_target_sz = target_sz;
    if prod(init_target_sz) > translation_model_max_area
        currentScaleFactor = sqrt(prod(init_target_sz) / translation_model_max_area);
    else
        currentScaleFactor = 1.0;
    end
    % target size at the initial scale初始大小
    base_target_sz = target_sz / currentScaleFactor;
    %window size, taking padding into account填充后的窗口大小
    sz = floor( base_target_sz * (1 + padding ));
    featureRatio = 4;
    output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
    use_sz = floor(sz/featureRatio);
    yf = single(fft2(gaussian_shaped_labels(output_sigma, use_sz)));
    interp_sz = base_target_sz * (1+CN_padding);% 插值后的响应图大小
    cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );

    if nScales > 0
        scale_sigma = nScalesInterp * scale_sigma_factor;
        
        scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
        scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);
        
        interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
        interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
        
        scaleSizeFactors = scale_step .^ scale_exp;
        interpScaleFactors = scale_step .^ interp_scale_exp_shift;
        
        ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
        ysf = single(fft(ys));
        scale_window = single(hann(size(ysf,2)))';
        
        %make sure the scale model is not to large, to save computation time
        if scale_model_factor^2 * prod(init_target_sz) > scale_model_max_area
            scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
        end
        
        %set the scale model size
        scale_model_sz = floor(init_target_sz * scale_model_factor);
        
        try
            im = imread([video_path  img_files{1}]);
        catch
            try
                im = imread([img_files{1}]);
            catch
                im = imread([video_path '/' img_files{1}]);
            end
        end
    
        %im = imread([video_path s_frames{1}]);
        
        %force reasonable scale changes
        min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
        max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
        
        max_scale_dim = strcmp(params.fDSST.s_num_compressed_dim,'MAX');
        if max_scale_dim
            s_num_compressed_dim = length(scaleSizeFactors);
        else
            s_num_compressed_dim = params.fDSST.s_num_compressed_dim;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 计算精度
    positions = zeros(numel(img_files), 4);
    boxes = zeros(numel(img_files), 4);
    apces = zeros(numel(img_files), 3);
    psrs = zeros(numel(img_files), 3);
    max_responses = zeros(numel(img_files), 3);
    % 初始化投影矩阵
    CN_projection_matrix = [];
    projection_matrix = [];
    % 计算fps
    time = 0;
%% 模型训练与更新
    for frame = 1:num_frames
        % 加载图像
        im = imread([video_path img_files{frame}]);

        tic;
        %% 检测
        if frame > 1
            % 计算压缩学习外观
            zp = feature_projection_CN(z_npca, z_pca, CN_projection_matrix, CN_cos_window);
            % 提取局部图像补丁的特征图
            [xo_npca, xo_pca] = get_subwindow_CN(im, pos, CN_sz, CN_non_compressed_features, CN_compressed_features, w2c);
            % 做降维和窗口化
            x = feature_projection_CN(xo_npca, xo_pca, CN_projection_matrix, CN_cos_window);
            % 计算分类器的响应
            kf = fft2(dense_gauss_kernel(CN_sigma, x, zp));
            CN_response = real(ifft2(alphaf_num .* kf ./ alphaf_den));
            %----------------------------------------------------------------
            [xt_npca, xt_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window);
            xtf = fft2(xt);% Zt
            fDSST_responsef = sum(hf_num .* xtf, 3) ./ (hf_den + lambda);% hf_num:At hf_den:Bt
            if interpolate_response > 0
                if interpolate_response == 2
                    % use dynamic interp size使用动态插值大小
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end 
                fDSST_responsef = resizeDFT2(fDSST_responsef, interp_sz);
            end
            fDSST_response = ifft2(fDSST_responsef, 'symmetric');
            
            apces(frame, 1) = APCE(CN_response);
            apces(frame, 2) = APCE(fDSST_response);

            Fusion_factor = apces(frame, 1) / (apces(frame, 1) + apces(frame, 2));
            response = Fusion_factor* CN_response + (1-Fusion_factor) * fDSST_response;
            
            
            apces(frame, 3) = APCE(response);

            psrs(frame, 1) = PSR(CN_response);
            psrs(frame, 2) = PSR(fDSST_response);
            psrs(frame, 3) = PSR(response);
            
            max_responses(frame, 1) = max(CN_response(:));
            max_responses(frame, 2) = max(fDSST_response(:));
            max_responses(frame, 3) = max(response(:));

            % 更新目标位置
%             [row, col] = find(CN_response == max(CN_response(:)), 1);
            [row, col] = find(response == max(response(:)), 1);
            disp_row = mod(row - 1 + floor((CN_sz(1)-1)/2), CN_sz(1)) - floor((CN_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((CN_sz(2)-1)/2), CN_sz(2)) - floor((CN_sz(2)-1)/2);
            pos = pos + [disp_row disp_col];
    
                 %scale search
            if nScales > 0
                
                %create a new feature projection matrix
                [xs_pca, xs_npca] = get_scale_subwindow(im,pos,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
                
                xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
                xsf = fft(xs,[],2);
                
                scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);
                
                interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
                
                recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
            
                %set the scale
                currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
                %adjust to make sure we are not to large or to small
                if currentScaleFactor < min_scale_factor
                    currentScaleFactor = min_scale_factor;
                elseif currentScaleFactor > max_scale_factor
                    currentScaleFactor = max_scale_factor;
                end
            end

        end

        %% 训练
        % 提取局部图像补丁的特征图训练分类器(CN)
        [xo_npca, xo_pca] = get_subwindow_CN(im, pos, CN_sz, CN_non_compressed_features, CN_compressed_features, w2c);
        %Compute coefficients for the tranlsation filter计算平移滤波器的系数
        [xl_npca, xl_pca] = get_subwindow(im, pos, sz, currentScaleFactor);

        if frame == 1
            % 初始化外观
            z_npca = xo_npca;
            z_pca = xo_pca;
            % 如果太多，则将压缩维数设置为最大值
            CN_num_compressed_dim = min(CN_num_compressed_dim, size(xo_pca, 2));
            %-------------------------------------------------------------
            h_num_pca = xl_pca;
            h_num_npca = xl_npca;
            num_compressed_dim = min(num_compressed_dim, size(xl_pca, 2));
        else
            % 更新外观
            z_npca = (1 - CN_learning_rate) * z_npca + CN_learning_rate * xo_npca;
            z_pca = (1 - CN_learning_rate) * z_pca + CN_learning_rate * xo_pca;
            %--------------------------------------------------------------
            h_num_pca = (1 - interp_factor) * h_num_pca + interp_factor * xl_pca;
            h_num_npca = (1 - interp_factor) * h_num_npca + interp_factor * xl_npca;
        end

        % 如果使用降维：更新投影矩阵
        if use_dimensionality_reduction
            % 计算平均外观
            data_mean = mean(z_pca, 1);

            % 从外观中减去平均值得到数据矩阵
            data_matrix = bsxfun(@minus, z_pca, data_mean);

            % 计算协方差矩阵
            cov_matrix = 1 / (prod(CN_sz) - 1) * (data_matrix' * data_matrix);

            % 计算主成分（pca_basis）和相应的方差
            if frame == 1
                [CN_pca_basis, CN_pca_variances, ~] = svd(cov_matrix);
            else
                [CN_pca_basis, CN_pca_variances, ~] = svd((1 - CN_compression_learning_rate) * CN_old_cov_matrix + CN_compression_learning_rate * cov_matrix);
            end

            % 计算投影矩阵作为第一主成分，并提取其对应的方差
            CN_projection_matrix = CN_pca_basis(:, 1:CN_num_compressed_dim);
            CN_projection_variances = CN_pca_variances(1:CN_num_compressed_dim, 1:CN_num_compressed_dim);

            if frame == 1
                % 使用计算的投影矩阵和方差初始化旧的协方差矩阵
                CN_old_cov_matrix = CN_projection_matrix * CN_projection_variances * CN_projection_matrix';
            else
                % 使用计算的投影矩阵和方差更新旧的协方差矩阵
                CN_old_cov_matrix = (1 - CN_compression_learning_rate) * CN_old_cov_matrix + CN_compression_learning_rate * (CN_projection_matrix * CN_projection_variances * CN_projection_matrix');
            end

        end
        % 使用新的投影矩阵投影新外观示例的特征
        x = feature_projection_CN(xo_npca, xo_pca, CN_projection_matrix, CN_cos_window);
        % 计算新的分类器系数
        kf = fft2(dense_gauss_kernel(CN_sigma, x));
        new_alphaf_num = CN_yf .* kf;
        new_alphaf_den = kf .* (kf + CN_lambda);
        %-----------------------------------------------------------
        data_matrix = h_num_pca;
        [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
        projection_matrix = pca_basis(:, 1:num_compressed_dim);
        % 将hog和gray组合特征由32维压缩为18维
        hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
        hf_num = bsxfun(@times, yf, conj(hf_proj));%At
        xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
        new_hf_den = sum(xlf .* conj(xlf), 3);%Bt

        if frame == 1
            % 第一帧，用单个图像训练
            alphaf_num = new_alphaf_num;
            alphaf_den = new_alphaf_den;
            %-------------------------------------------------------
            hf_den = new_hf_den;
        else
            % 后续帧，更新模型
            alphaf_num = (1 - CN_learning_rate) * alphaf_num + CN_learning_rate * new_alphaf_num;
            alphaf_den = (1 - CN_learning_rate) * alphaf_den + CN_learning_rate * new_alphaf_den;
            %-------------------------------------------------------
            hf_den = (1 - interp_factor) * hf_den + interp_factor * new_hf_den;
        end
        %Compute coefficents for the scale filter
        if nScales > 0
            
            %create a new feature projection matrix
            % 提取17个不同尺度的patch再统一resize到scale_model_size大小，然后再分别提取这些patch的hog特征，将hog特征拉成一维的
            % 最后输出的xs_pca是17个一维的hog向量，用于后面的降维，后面的那个变量没用，是空的
            % base_target_sz：目标的基准大小
            % scale_model_size：尺度模型的基准大小，保证其面积小于512，如果大于则宽高等比例的放缩
            [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
            
            if frame == 1
                s_num = xs_pca;
            else
                s_num = (1 - interp_factor) * s_num + interp_factor * xs_pca;
            end
            
            bigY = s_num;
            bigY_den = xs_pca;
            
            if max_scale_dim
                [scale_basis, ~] = qr(bigY, 0);
                [scale_basis_den, ~] = qr(bigY_den, 0);
            else
                [U,~,~] = svd(bigY,'econ');
                scale_basis = U(:,1:s_num_compressed_dim);
            end
            scale_basis = scale_basis';
            
            %create the filter update coefficients
            sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
            sf_num = bsxfun(@times,ysf,conj(sf_proj));% scale:At
            
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
            xsf = fft(xs,[],2);
            new_sf_den = sum(xsf .* conj(xsf),1);% scale:Bt
            
            if frame == 1
                sf_den = new_sf_den;
            else
                sf_den = (1 - interp_factor) * sf_den + interp_factor * new_sf_den;
            end
        end
        target_sz = floor(base_target_sz * currentScaleFactor);

        %% 结果与可视化
        % 保存位置
        positions(frame, :) = [pos target_sz];
        time = time + toc;
        rect_position = [pos([2, 1]) - target_sz([2, 1]) / 2, target_sz([2, 1])];
        boxes(frame, :) = rect_position;
        % 可视化
        if visualization == 1

            if frame == 1 %first frame, create GUI
                figure('Name', ['Tracker - ' video_path]);
                im_handle = imshow(uint8(im), 'Border', 'tight', 'InitialMag', 100 + 100 * (length(im) < 500));
                rect_handle = rectangle('Position', rect_position, 'EdgeColor', 'g');
                text_handle = text(10, 10, int2str(frame));
                set(text_handle, 'color', [0 1 1]);
            else

                try %subsequent frames, update GUI
                    set(im_handle, 'CData', im)
                    set(rect_handle, 'Position', rect_position)
                    set(text_handle, 'string', int2str(frame));
                catch
                    return
                end

            end

            drawnow
        end

    end

    fps = num_frames / time;
    results.positions = positions;
    results.boxes = boxes;
    results.fps = fps;

    if draw_apce
        figure()
        subplot(3,1,1)
        plot((1:num_frames), apces(:,1))
        subplot(3,1,2)
        plot((1:num_frames), apces(:,2))
        subplot(3,1,3)
        plot((1:num_frames), apces(:,3))
        title("apce")
    
        figure()
        subplot(3,1,1)
        plot((1:num_frames), psrs(:,1))
        subplot(3,1,2)
        plot((1:num_frames), psrs(:,2))
        subplot(3,1,3)
        plot((1:num_frames), psrs(:,3))
        title("psr")
    
        figure()
        subplot(3,1,1)
        plot((1:num_frames), max_responses(:,1))
        subplot(3,1,2)
        plot((1:num_frames), max_responses(:,2))
        subplot(3,1,3)
        plot((1:num_frames), max_responses(:,3))
        title("max_response")
    end
end