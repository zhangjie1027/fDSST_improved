function labels = gaussian_shaped_labels(sigma, sz)
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2)); %二维高斯分布，峰值位于中心点处
	labels = circshift(labels, -floor(sz(1:2) / 2) + 1); %circshift - 循环平移数组 %峰值点移动到左上角
    assert(labels(1,1) == 1) 
end