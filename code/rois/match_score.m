% Compute a match score based on the spatial and temporal trace profiles.
%
% Assumes that the information has been already properly rescaled.
%
% Args:
% spacea - spatial profile of the first source
% maska - mask of the first source, i.e. (spacea ~= 0)
% timea - temporal trace profile of the first source
% spaceb - spatial profile of the second source
% maskb - mask of the second source, i.e. (spaceb ~= 0)
% timeb - temporal trace profile of the second source
% params - rois_params() defining the constants for matching
%
% Returns:
% score - the match score
function score = match_score(spacea, maska, timea, spaceb, maskb, timeb, params)

	p_limit_common = params.combine_far_min_common;

	mask_common = maska .* maskb;
	count_common = sum(mask_common, "all");

	% if no overlap, can't merge
	if count_common == 0
		score = 0;
		return;
	end

	% find the least-squares fits
	lsq_a2b = fista_fit(spacea, spaceb, params);
	lsq_b2a = fista_fit(spaceb, spacea, params);
	lsq_a2b_common = fista_fit(spacea .* mask_common, spaceb .* mask_common, params);
	lsq_b2a_common = fista_fit(spaceb .* mask_common, spacea .* mask_common, params);
	% Some combinations produce copies of values, since they have an equivalent meaning:
	% (lsq_b2a_omask = fista_fit(spaceb .* ra.mask, spacea .* ra.mask, params)) == lsq_b2a_common
	% (lsq_b2a_nmask = fista_fit(spaceb .* rb.mask, spacea .* rb.mask, params)) == lsq_b2a
	% (lsq_a2b_omask = fista_fit(spacea .* ra.mask, spaceb .* ra.mask, params)) == lsq_a2b
	% (lsq_a2b_nmask = fista_fit(spacea .* rb.mask, spaceb .* rb.mask, params)) == lsq_a2b_common

	% high values of these ratios (which are always <= 1) mean that one ROI
	% matches fits well inside of the other ROI
	a2b_ratio = lsq_a2b/lsq_a2b_common;
	b2a_ratio = lsq_b2a/lsq_b2a_common;

	% --- find the mutual fits of activation traces

	tmask_common = (timea .* timeb) > 0;
	tcount_common = sum(tmask_common, "all");

	if tcount_common > 0
		% find the least-squares fits
		plsq_a2b = fista_fit(timea, timeb, params);
		plsq_b2a = fista_fit(timeb, timea, params);
		plsq_a2b_common = fista_fit(timea .* tmask_common, timeb .* tmask_common, params);
		plsq_b2a_common = fista_fit(timeb .* tmask_common, timea .* tmask_common, params);

		% high values of these ratios (which are always <= 1) mean that one ROI
		% matches fits well inside of the other ROI
		pa2b_ratio = plsq_a2b/plsq_a2b_common;
		pb2a_ratio = plsq_b2a/plsq_b2a_common;
	else
		pa2b_ratio = 0;
		pb2a_ratio = 0;
	end

	% --- make decisions

	min_score_px = min(a2b_ratio, b2a_ratio);
	min_score_trace = min(pa2b_ratio, pb2a_ratio);

	max_score_px = max(a2b_ratio, b2a_ratio);
	max_score_trace = max(pa2b_ratio, pb2a_ratio);

	% the symmetrically good fit in both spatial and temporal ways
	% is the best kind because it means that both ROIs are close
	if min_score_px >= p_limit_common && min_score_trace >= p_limit_common
		score = 4 + min_score_px * min_score_trace;
		return;
	end

	% symmetrially good fit in one way only, spatial or temporal,
	% and asymmetrically good in the other way
	if min_score_px >= p_limit_common && max_score_trace >= p_limit_common ...
	|| max_score_px >= p_limit_common && min_score_trace >= p_limit_common
		score = 3 + max(min_score_px * max_score_trace, max_score_px * min_score_trace);
		return;
	end

	% asymmetrically good fit in both ways
	if max_score_px >= p_limit_common && max_score_trace >= p_limit_common
		score = 2 + max_score_px * max_score_trace;
		return;
	end

	% asymmetrically good fit in at least space
	if max_score_px >= p_limit_common
		score = 1 + max_score_px;
		return;
	end

	% last resort
	score = min(min_score_px, min_score_trace);
end
