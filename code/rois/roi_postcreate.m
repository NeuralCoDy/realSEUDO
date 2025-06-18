% Fill in the parts of a ROI that are derived from the other parts
% (maybe it should be a class and a class method?)
%
% ROI fields are described in find_still_rois_in().
%
% Args:
% r - ROI with mask, count, and raw already filled
function r = roi_postcreate(r)
	% sorted pixels within the mask
	r.sorted = sort(reshape(r.raw, 1, []));
	r.sorted = r.sorted(1, end-r.count+1:end);
	% and then SEUDO normalizes the ROIs to add up to 1
	r.normalize_coeff1 = sum(r.raw, "all");
    if r.normalize_coeff1 == 0
        r.normalize_coeff1 = 1;
    end
	r.normalized1 = r.raw / r.normalize_coeff1;
	% and then for detection of ROIS, they get normalized once more;
	% this can also be written with the same result as
	%   r.normalize_coeff2 = sqrt(sum(r.raw(:).^2));
	% but the 2-stage computation more obviously follows estimateTimeCoursesWithSEUDO()
	r.normalize_coeff2 = r.normalize_coeff1 * sqrt(sum(r.normalized1(:).^2));
    if r.normalize_coeff2 == 0
        r.normalize_coeff2 = 1;
    end
	r.normalized2 = r.raw / r.normalize_coeff2;

	% estimate the brightness as going one standard deviation up from mean
	r.brightness = mean(r.sorted) + std(r.sorted);
	r.normalize_coeff3 = r.brightness; % an alias
	r.normalized3 = r.raw / r.normalize_coeff3;

	% Because r.normalize_coeff1 = sum(r.raw, "all") and r.normalized2 = r.raw / r.normalize_coeff2,
	% the expression for r.total is equivalent to sum(r.normalized2, "all");
	r.total = r.normalize_coeff1 / r.normalize_coeff2;
	r.last_weight = 0;

	% find the bounding box
	ind = find(sum(r.mask, 1));
	if length(ind) < 1
		r.bbox.col_min = 0;
		r.bbox.col_max = 0;
		r.bbox.row_min = 0;
		r.bbox.row_max = 0;
		r.bboxed = [];
	else
		r.bbox.col_min = ind(1);
		r.bbox.col_max = ind(end);
		ind = find(sum(r.mask, 2));
		r.bbox.row_min = ind(1);
		r.bbox.row_max = ind(end);

		% convenience values for easy examination
		r.bboxed = r.raw(r.bbox.row_min:r.bbox.row_max, r.bbox.col_min:r.bbox.col_max);
	end

	r.bbox_direct = [r.bbox.row_min r.bbox.row_max r.bbox.col_min r.bbox.col_max];

	% no trace yet
	r.trace = nan;
end
