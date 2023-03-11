function apce = APCE(rsp)
    max_val = max(rsp(:));
    min_val = min(rsp(:));
    sub_mat = (rsp - min_val).^2;
    apce = (max_val - min_val)^2 / (sum(sub_mat(:)) / numel(rsp));
end