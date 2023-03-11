function psr = PSR(rsp)
%     psr = (max(rsp(:)) - mean(rsp(:)))/std(rsp(:));
    max_val = max(rsp(:));
    u_s = (sum(rsp(:)) - max_val) / (numel(rsp)-1);
    sub_mat = (rsp - u_s).^2;
    std_s = (sum(sub_mat(:)) - (max_val - u_s)^2) / (numel(rsp)-1);
    psr = (max_val - u_s) / std_s;
end