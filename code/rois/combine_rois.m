
% Combine two sets of ROIs by folding the newly found ROIs into the
% pre-existing set.
%
% Args:
% rois - old set to be extended
% new_rois - new set to be folded in
% mode - a string selecting the mode for combining:
%   "near": the old set is temporally near to the new one;
%   "far": the old set is temporally far from the new one;
%     the criteria for merging are different for combining the nearby
%     frames (where the ROIs can be observed growing and shrinking)
%     and for combining the far-away frames where we expect to observe
%     the ROIs at full brightness that they have achieved during an
%     activation.
%   "far_add_only": do only the addition, no subtraction, in far mode
%   "score_matrix": doesn't combine anything but returns a matrix of
%     match score of each rois to each new_rois, using normalized2 values;
%     the result matrix has values for rois as index 1, new_rois as index 2
% params - controlling parameters
%
% Results:
% rois - the combined set
function rois = combine_rois(rois, new_rois, mode, params)
	if mode == "near"
		rois = combine_rois_near(rois, new_rois, params);
	elseif mode == "far"
		rois = combine_rois_far(rois, new_rois, 0, params);
	elseif mode == "far_add_only"
		rois = combine_rois_far(rois, new_rois, 1, params);
	elseif mode == "test_score_far"
		% test mode, compute the overlap score
		[score, detail] = overlap_score_far(rois(1), new_rois(1), 1, 2, params);
		rois = [string(score), detail.opcode]
	elseif mode == "test_subtract"
		% test mode, subtract the second ROI from first
		[rois, sr, success] = subtract_two(rois(1), new_rois(1), 1, 2, params);
		rois = [rois, sr];
		success
	elseif mode == "score_matrix"
		% compute match score of each-to-each
		mat = zeros(length(rois), length(new_rois));
		for i=1:length(rois)
			for j = 1:length(new_rois)
				r1 = rois(i);
				r1.raw = r1.normalized2;
				r2 = new_rois(j);
				r2.raw = r2.normalized2;
				[score, detail] = overlap_score_far(r1, r2, 1, 2, params);
				mat(i, j) = score;
			end
		end
		rois = mat; % return the result
	else
		error("Unknown mode ("+ string(mode)+ ")")
	end
end

% The near logic of combine_rois().
function rois = combine_rois_near(rois, new_rois, params)
	for ni = 1:length(new_rois)
		nr = new_rois(ni);

		% old ROIs that are overlapping
		overlap = find_overlap(rois, nr);

		% an example for studying the merging:
		%   [r, recent] = find_movie_rois([], [], M, 11401, 11500, 2, 30, 0.2, 1)
		%   r2 = find_movie_rois(r, recent, M, 34362, 34368, 2, 50, 0.2, 1)

		if length(overlap) > 1
			LOG("DEBUG near ROI merging, frame="+ string(nr.frame_id)+ "  -- "+ join(string([ rois(overlap).frame_id ])) );
			% nr.bboxed
		end

		% index of the ROI that is a duplicate of nr.
		duplicate = nan;
		newi = length(rois);

		% duplicates that will be removed
		to_remove = [];

		for ii = 1:length(overlap)
			oi = overlap(ii);
			or = rois(oi);

			[xr, success] = try_combine_two_near(or, nr, oi, newi, params);

			if success
				LOG("DEBUG near ROI "+ string(oi)+ " is growing on frame="+ string(xr.frame_id));

				if params.online && sum(xr.raw ~= or.raw, 'all') ~= 0
					if params.norm_mode == 0
						fit_old = fista_fit(xr.normalized2, or.normalized2, params);
						fit_new = fista_fit(xr.normalized2, nr.normalized2, params);
						% make an event only if something previously reported has changed
						if abs(fit_old - 1) > params.min_detect_early_mod ...
						|| (abs(fit_new - 1) > params.min_detect_early_mod && nr.stable_id >= 0)
							rois_event("mod_merge", xr.event_frame_id, xr.stable_id, fit_old, nr.stable_id, fit_new);
						end
					elseif params.norm_mode == 1
						% make an event only if something previously reported has changed
						if xr.total - or.total > or.total * params.min_detect_early_mod
							rois_event("mod_merge", xr.event_frame_id, xr.stable_id, or.total / xr.total, nr.stable_id, nr.total / xr.total);
						end
					elseif params.norm_mode == 2
						if abs(xr.brightness - or.brightness) > or.brightness * params.min_detect_early_mod
							rois_event("mod_merge", xr.event_frame_id, xr.stable_id, ...
								or.brightness / xr.brightness, nr.stable_id, nr.brightness / xr.brightness);
						end
					end
				end

				rois(oi) = xr;
				% xr.bboxed

				if ~isnan(duplicate)
					LOG("DEBUG near ROI "+ string(duplicate)+ " disappeared on frame="+ string(nr.frame_id)+ " by combining with "+ string(oi));
					to_remove = [to_remove duplicate];
				end

				duplicate = oi;
				newi = oi;
				nr = xr;
			end
		end

		if isnan(duplicate)
			% don't assign the stable id yet
			LOG("DEBUG near ROI "+ string(length(rois)+1)+ " (stable "+ string(nr.stable_id)+ ") added on frame="+ string(nr.frame_id));
			rois = [rois nr];
		end

		% in descending order, so that removed duplicates won't mess up the numbering of other duplicates
		to_remove = sort(to_remove, 'descend');
		for ii = 1:length(to_remove)
			duplicate = to_remove(ii);
			LOG("DEBUG near ROI "+ string(duplicate)+ " (stable "+ string(rois(duplicate).stable_id)+ ") removed on frame "+ string(nr.frame_id));
			rois = [rois(1:(duplicate-1)) rois((duplicate+1):end)];
		end
	end
end

% Try to combine two temporally-near ROIs
%
% Args:
% or - old (previously seen) ROI to combine
% nr - new (newly detected) ROI to combine
% oi - index of old ROI
% ni - index of new ROI
% params - controlling parameters
%
% Results:
% xr - combined ROI on success, [] on failure
% success - 1 if the combination can be done, 0 if not
function [xr, success] = try_combine_two_near(or, nr, oi, ni, params)
	% parameters that control the logic
	p_min_common = params.combine_near_min_common;

	mask_common = nr.mask .* or.mask;
	count_common = sum( reshape( mask_common, 1, []) );

	mask_removed = (or.mask - mask_common) > 0;
	count_removed = sum( reshape( mask_removed, 1, []) );

	mask_added = (nr.mask - mask_common) > 0;
	count_added = sum( reshape( mask_added, 1, []) );

	if count_common == 0
		% completely not overlapping, can't combine
		xr = [];
		success = 0;
		return;
	end

	n_semi_perimeter = (nr.bbox.row_max - nr.bbox.row_min + 1 ...
		+ nr.bbox.col_max - nr.bbox.col_min + 1) / 2;
	o_semi_perimeter = (or.bbox.row_max - or.bbox.row_min + 1 ...
		+ or.bbox.col_max - or.bbox.col_min + 1) / 2;

	if count_added <= n_semi_perimeter ...
	|| count_removed <= o_semi_perimeter ...
	|| (count_common >= or.count * p_min_common ...
		&& count_common >= nr.count * p_min_common)
		% or
		% nr
		% or.bboxed
		% or.bbox_direct
		% nr.bboxed
		% nr.bbox_direct
		% count_common
		% count_removed
		% count_added
		% o_semi_perimeter
		% n_semi_perimeter

		xr = combine_two(or, nr, oi, ni);

		success = 1;
		return;
	end

	% there is too much difference, cannot be combined
	% count_common
	% count_removed
	% count_added
	xr = [];
	success = 0;
end

% The far logic of combine_rois().
%
% an example for studying the merging:
%   [r, recent] = find_movie_rois([], [], M, 11401, 11500, 2, 30, 0.2, 1)
%   r2 = find_movie_rois(r, recent, M, 34362, 34368, 2, 50, 0.2, 1)
function rois = combine_rois_far(rois, new_rois, add_only, params)
	while length(new_rois) > 0
		nr = new_rois(1);
		new_rois = new_rois(2:end);

		if nr.id > 0
			% if we've got a reference to a previously known ROI, it might
			% have been updated while this copy has been waiting on the list;
			% get the most recent copy (this would have been much easier if
			% Matlab had proper references, then new_rois would just contain
			% references)
			nr = rois(nr.id);
		end

		if 0 % try to parallelize the search
			candidates = [];

			parfor oi = 1:length(rois)
				if oi == nr.id
					% don't match to self
					continue;
				end

				or = rois(oi);

				if or.bbox.col_min > nr.bbox.col_max ...
				|| nr.bbox.col_min > or.bbox.col_max ...
				|| or.bbox.row_min > nr.bbox.row_max ...
				|| nr.bbox.row_min > or.bbox.row_max ...
					continue;
				end

				[score, ops] = overlap_score_far(rois(oi), nr, oi, nr.id, params);
				cand = struct('idx', oi, 'score', score, 'ops', ops);
				candidates = [candidates cand];
			end

			if ~isempty(candidates)
				[score, cand_idx] = max([candidates(:).score]);
				oi = candidates(cand_idx).idx;
				ops = candidates(cand_idx).ops;
			else
				oi = -1;
				score = -1;
				ops.opcode = "";
			end

		else % sequential search

			% old ROIs that are overlapping
			overlap = find_overlap(rois, nr);

			% find the overlap with highest score
			oi = -1;
			score = -1;
			ops.opcode = "";
			for ii = 1:length(overlap)
				cand_oi = overlap(ii);
				if cand_oi == nr.id
					continue
				end

				[cand_score, cand_ops] = overlap_score_far(rois(cand_oi), nr, cand_oi, nr.id, params);
				if cand_score > score
					score = cand_score;
					ops = cand_ops;
					oi = cand_oi;
				end
			end
		end

		if ops.opcode == "o+n"
			[rois, new_rois] = combine_two_far(rois, new_rois, oi, nr, params);
		elseif ~add_only && ops.opcode == "o-n"
			or = rois(oi);
			LOG("DEBUG subtracting ROI "+ string(nr.id)+ " from ROI "+ string(or.id));
			[res, nr, success] = subtract_two(or, nr, or.id, nr.id, params);

			if success
				% replace the old ROI with the first element of result, merge_trace is already processed
				LOG("DEBUG ROI "+ string(or.id)+ " shrunk on frame="+ string(nr.frame_id)+ " on subtracting of "+ string(nr.id));
				res(1).id = or.id;
				rois(or.id) = res(1);

				if length(res) > 1
					LOG("DEBUG ROI "+ string(or.id)+ " got sharded on frame="+ string(nr.frame_id)+ " on subtracting of "+ string(nr.id));
				end

				% reprocess the remaining shards and the new ROI afterwards;
				% the first shard keeps the position of the split ROI, the rest are treated as new
				new_rois = [res nr new_rois];
			else
				% subtraction doesn't work out well, merge instead
				[rois, new_rois] = combine_two_far(rois, new_rois, oi, nr, params);
			end
		elseif ~add_only && ops.opcode == "n-o"
			or = rois(oi);
			LOG("DEBUG subtracting ROI "+ string(or.id)+ " from ROI "+ string(nr.id));
			[res, or, success] = subtract_two(nr, or, nr.id, or.id, params);

			if success
				rois(oi) = or;
				if (nr.id > 0)
					% replace the new ROI with the first element of result, merge_trace is already processed
					LOG("DEBUG ROI "+ string(nr.id)+ " shrunk on frame="+ string(nr.frame_id)+ " on subtracting of "+ string(or.id));
					res(1).id = nr.id;
					rois(nr.id) = res(1);

					if length(res) > 1
						LOG("DEBUG ROI "+ string(nr.id)+ " got sharded on frame="+ string(nr.frame_id)+ " on subtracting of "+ string(or.id));
					end
				end
				% reprocess the remaining shards;
				new_rois = [res new_rois];
			else
				% subtraction doesn't work out well, merge instead
				[rois, new_rois] = combine_two_far(rois, new_rois, oi, nr, params);
			end
		else
			if nr.id <= 0
				% ROI is new and can't merge with anything, add as a new ROI
				rois = add_roi_far(rois, nr);
			end
		end
	end
end

% Compute the overlap score two temporally-far ROIs
%
% Args:
% or - old (previously seen) ROI to potentially combine
% nr - new (newly detected) ROI to potentially combine
% oi - index of old ROI
% ni - index of new ROI
% params - controlling parameters
%
% Results:
% score - an estimation of how the two ROIs fit together, greater is better,
%   and the values are chosen to prefer the combination over subtraction
% details - operation details for the combination, a structure with fields:
%   opcode: "" (means can't combined), "o+n", "o-n", "n-o"
function [score, details] = overlap_score_far(or, nr, oi, ni, params)
	% parameters that control the logic
	% ---
	% if the old and new ROI have this much in common, unify them,
	% the same criteria is used for subtraction but when the relation
	% between ROIs is asymmetric, one side being above and the other below
	p_limit_common = params.combine_far_min_common;
	% when subtracting, maximum ratio of brightness of larger ROI to smaller one
	p_max_brightness_ratio = params.subtract_far_max_brightness_ratio;
	% when subtracting, maximum value representing the symmetry of fitting
	% one side to the other (i.e. the fitting has to have a high enough asymmetry)
	p_subtract_max_symmetry = params.subtract_far_max_symmetry;
	% ---

	mask_common = nr.mask .* or.mask;
	count_common = sum( reshape( mask_common, 1, []) );

	count_removed = or.count - count_common;
	count_added = nr.count - count_common;

	% if no overlap, can't merge
	if count_common == 0
		score = 0;
		details.opcode = "";
		return;
	end

	% find the least-squares fits with FISTA
	lsq_o2n = fista_fit(or.raw, nr.raw, params);
	lsq_n2o = fista_fit(nr.raw, or.raw, params);
	lsq_o2n_common = fista_fit(or.raw .* mask_common, nr.raw .* mask_common, params);
	lsq_n2o_common = fista_fit(nr.raw .* mask_common, or.raw .* mask_common, params);
	% Some combinations produce copies of values, since they have an equivalent meaning:
	% (lsq_n2o_omask = fista_fit(nr.raw .* or.mask, or.raw .* or.mask, params)) == lsq_n2o_common
	% (lsq_n2o_nmask = fista_fit(nr.raw .* nr.mask, or.raw .* nr.mask, params)) == lsq_n2o
	% (lsq_o2n_omask = fista_fit(or.raw .* or.mask, nr.raw .* or.mask, params)) == lsq_o2n
	% (lsq_o2n_nmask = fista_fit(or.raw .* nr.mask, nr.raw .* nr.mask, params)) == lsq_o2n_common

	% high values of these ratios (which are always <= 1) mean that one ROI
	% matches fits well inside of the other ROI
	o2n_ratio = lsq_o2n/lsq_o2n_common;
	n2o_ratio = lsq_n2o/lsq_n2o_common;

	% the symmetrically good fit means that we can merge, because both ROIs are close
	score = min(o2n_ratio, n2o_ratio);
	if score >= p_limit_common
		score = 2 + score;
		details.opcode = "o+n";
		return;
	end

	% the asymmetrical good fit of one side means that the smaller ROI can be subtracted from the larger one
	if o2n_ratio >= p_limit_common
		if lsq_o2n_common >= p_max_brightness_ratio
			% one is much brighter than the other, the pale one should be merged instead
			score = 2 + min(o2n_ratio, n2o_ratio);
			details.opcode = "o+n";
			return;
		elseif n2o_ratio / o2n_ratio <= p_subtract_max_symmetry
			score = 1 + o2n_ratio - n2o_ratio;
			details.opcode = "n-o";

			% count_common
			% count_removed
			% count_added

			return;
		end
	elseif n2o_ratio >= p_limit_common
		if lsq_n2o_common >= p_max_brightness_ratio
			% one is much brighter than the other, the pale one should be merged instead
			score = 2 + min(o2n_ratio, n2o_ratio);
			details.opcode = "o+n";
			return;
		elseif o2n_ratio / n2o_ratio <= p_subtract_max_symmetry
			score = 1 + n2o_ratio - o2n_ratio;
			details.opcode = "o-n";

			% count_common
			% count_removed
			% count_added

			return;
		end
	end

	details.opcode = "";
	return;
end

% Combine two ROIs after they have been decided to be combinable.
%
% Args:
% or - old (previously seen) ROI to combine
% nr - new (newly detected) ROI to combine
% oi - index of old ROI
% ni - index of new ROI
%
% Results:
% xr - combined ROI
function xr = combine_two(or, nr, oi, ni)
	LOG("DEBUG ROI is combining from frame="+ string(or.frame_id)+ " and frame="+ string(nr.frame_id));

	xr = struct(nr); % make a copy
	xr.stable_id = or.stable_id; % the old ROI is more stable

	% there are multiple ways to combine frames, taking the maximum should be good enough
	xr.frame = max(or.frame, nr.frame);

	xr.event_frame_id = max(or.event_frame_id, nr.event_frame_id);

	if size(nr.frame_id_trace, 1) == 1 && nr.frame_id_trace(1,1) == nr.frame_id
		% new ROI consists of one frame
		xr.frame_id_trace = or.frame_id_trace;
		if xr.frame_id_trace(end, 2) == nr.frame_id - 1
			xr.frame_id_trace(end, 2) = nr.frame_id;
		else 
			xr.frame_id_trace = [xr.frame_id_trace; nr.frame_id nr.frame_id];
		end
	else
		% This may lead to duplicate ranges, but no big deal, close enough,
		% and this shows better the history of both branches.
		xr.frame_id_trace = [or.frame_id_trace ; nr.frame_id_trace];
	end

	% not very meaningful but something, cutoff isn't used much any more...
	xr.cutoff = min(or.cutoff, nr.cutoff);
	% (should xr.hist_der also be somehow updated?)

	xr.mask = (or.mask + nr.mask) > 0;
	xr.count = sum( reshape( xr.mask, 1, []) );
	% use the candidate mask to cut the values out of the frame
	xr.raw = double(xr.frame .* xr.mask);
	xr = roi_postcreate(xr);
end

% Combine two temporally far ROIs after they have been decided to be combinable,
% maintaining the merge_trace and indexing, removing the new ROI if it was
% on the list.
%
% Args:
% 
% rois - the list of known ROIs
% new_rois - the list of ROIs to be processed yet by combine_rois_far()
% oi - index of old (previously seen) ROI to combine, which will be replaced with the
%   combination
% nr - new (newly detected) ROI to combine, nr.id <= 0 if it's not on the rois list yet
% params - controlling parameters
%
% Results:
% rois - updated list of known ROIs
% new_rois - the list of ROIs to be processed yet by combine_rois_far(), it gets updated
%   by prepending the newly updated old ROI, and updating the indexes of ROIs as needed
%   for removal of the consumed new ROI
function [rois, new_rois] = combine_two_far(rois, new_rois, oi, nr, params)
	or = rois(oi);

	if nr.id > 0
		% copy without copying merge_trace to avoid exploding memory
		xor = struct(or);
		xor.merge_trace = {};
		xnr = struct(nr);
		xnr.merge_trace = {};
		merge_trace = [ { "<{" nr.frame_id nr.id 0}; nr.merge_trace; { "<}" nr.frame_id nr.id xnr } ; ...
			{ ">{" nr.frame_id or.id 0}; or.merge_trace; { ">}" nr.frame_id or.id xor } ];
	else
		% copy without copying merge_trace to avoid exploding memory
		xor = struct(or);
		xor.merge_trace = {};
		xnr = struct(nr);
		xnr.merge_trace = {};
		merge_trace = [or.merge_trace; { "<+" nr.frame_id -1 xnr }; { ">+" nr.frame_id or.id xor }]; 
	end
	xr = combine_two(or, nr, or.id, nr.id);
	if params.merge_trace
		xr.merge_trace = merge_trace;
	else
		xr.merge_trace = {};
	end

	if params.online
		if params.norm_mode == 0
			rois_event("mod_merge", xr.event_frame_id, xr.stable_id, fista_fit(xr.normalized2, or.normalized2, params), ...
				nr.stable_id, fista_fit(xr.normalized2, nr.normalized2, params));
		elseif params.norm_mode == 1
			rois_event("mod_merge", xr.event_frame_id, xr.stable_id, or.total / xr.total, ...
				nr.stable_id, nr.total / xr.total);
		elseif params.norm_mode == 2
			rois_event("mod_merge", xr.event_frame_id, xr.stable_id, or.brightness / xr.brightness, ...
				nr.stable_id, nr.brightness / xr.brightness);
		end
	end

	LOG("DEBUG ROI "+ string(or.id)+ " is growing on frame="+ string(xr.frame_id)+ " by absorbing "+ string(nr.id));

	xr.id = oi;
	rois(oi) = xr;
	new_rois = [xr new_rois];

	if nr.id > 0
		LOG("DEBUG ROI "+ string(nr.id)+ " disappeared on frame="+ string(nr.frame_id)+ " by combining with "+ string(or.id));
		[rois, new_rois] = remove_roi_at_idx(rois, new_rois, nr.id);
	end
end

% Subtract one ROI from another.
%
% Args:
% br - "big" ROI to subtract from
% sr - "small" ROI to be subtracted
% bi - index of big ROI
% si - index of small ROI
% params - controlling parameters
%
% Results:
% rois - what remained of the big ROI after subtraction, this is a list of
%   1 or more ROIs. Multiple ROIs may be produced if the
%   big ROI gets broken up into multiple disjointed regions.
%   (There is no way to absorb anything into the small ROI because we
%   don't know what part of the brightness of the overlapping region
%   came from the other cell).
% sr - the "small" ROI with possibly adjusted brightness (it can be adjusted only up)
% success - flag: 1 if the subtraction worked, 0 if not, either if the estimation that the
%   two are subtractable was overly optimistic and the result would still
%   be seen as an overlap (this generally happens when sr is not sufficiently
%   bright and too sparse) or if two ROIs are so close that after subtracting
%   them there is nothing substantial left.
function [rois, sr, success] = subtract_two(br, sr, bi, si, params)
	LOG("DEBUG ROI is subtracting from frame="+ string(br.frame_id)+ " the frame="+ string(sr.frame_id));

	rois = [];

	mask_common = br.mask .* sr.mask;
	lsq_s2b_common = fista_fit(sr.raw .* mask_common, br.raw .* mask_common, params);

	% copy without copying merge_trace to avoid exploding memory
	xbr = struct(br);
	xbr.merge_trace = {};
	xsr = struct(sr);
	xsr.merge_trace = {};
	if params.merge_trace
		common_merge_trace = [br.merge_trace; { "<-" sr.frame_id si xsr }; { ">-" br.frame_id bi xbr }]; 
	else
		common_merge_trace = {};
	end

	% consider in-frame only the old contents of the big ROI
	frame = br.raw - lsq_s2b_common * sr.raw;

	% now drop the small discontiguous regions

	% mostly copied from find_still_rois() {
		suspects = frame > br.cutoff;
		% do the blurring to join the disjointedness
		if br.mask_blur_rad > 0
			% maybe a square kernel would make more sense?
			kernel = fspecial('disk', br.mask_blur_rad);
			suspects = conv2(suspects, kernel, 'same') > 0;
		end
		% label the suspected ROIs
		[labels, n_labs] = bwlabel(suspects, 8);
	
		for ll = 1:n_labs
			% check if the candidate label is big enough
			mask = (labels == ll);
			count = sum( reshape( mask, 1, []) );

			if count >= br.min_roi_size
				% use the candidate mask to cut the values out of the frame
				raw = double(frame .* mask);
				% count the real pixels that are > 0 to avoid the images with
				% little real pixels
				rawcount = sum( reshape( raw > 0, 1, []) );
				rawsum = sum( reshape( raw, 1, []) );

				if rawcount >= br.min_roi_size && rawsum / rawcount >= br.min_avg_px
					r = struct(br); % make a copy
					r.id = -1;
					r.stable_id = rois_new_stable_id();
					r.frame_id = max(br.frame_id, sr.frame_id); % last frame that caused a change
					r.event_frame_id = max(br.event_frame_id, sr.event_frame_id);
					% frame_id_trace is unchanged, because nothing gets added - this is
					% a little inconsistent with frame_id but makes more sense
					r.merge_trace = common_merge_trace;

					r.count = count;

					r.mask = mask;
					r.raw = raw;
					r.subtracted = 1;
					r = roi_postcreate(r);

					% append this ROI to the result
					rois = [rois r];
				end
			end
		end
	% }

	% now check for success

	if length(rois) == 0
		% two ROIs were so close that after subtracting there is nothing left
		success = 0;
		return;
	end

	for i = 1:length(rois)
		[cand_score, cand_ops] = overlap_score_far(rois(i), sr, -1, -1, params);
		if cand_ops.opcode ~= ""
			% SBXXX
			%show_rois([br, sr, rois], 0);
			%pause

			rois = [];
			success = 0;
			return;
		end
	end

	if lsq_s2b_common > 1
		sr.raw = sr.raw * lsq_s2b_common;
		sr.frame = sr.frame * lsq_s2b_common;
		% direct scaling doesn't change the normalized values
		sr.normalize_coeff1 = sr.normalize_coeff1 * lsq_s2b_common;
		sr.normalize_coeff2 = sr.normalize_coeff2 * lsq_s2b_common;
		sr.subtracted = 1;
	end

	if params.online
		rois(1).stable_id = br.stable_id;

		if params.norm_mode == 0
			rois_event("mod_subtract", rois(1).event_frame_id, rois(1).stable_id, fista_fit(rois(1).normalized2, br.normalized2, params), ...
				sr.stable_id, 0);
		elseif params.norm_mode == 1
			rois_event("mod_subtract", rois(1).event_frame_id, rois(1).stable_id, br.total / rois(1).total, ...
				sr.stable_id, 0);
		elseif params.norm_mode == 2
			rois_event("mod_subtract", rois(1).event_frame_id, rois(1).stable_id, br.brightness / rois(1).brightness, ...
				sr.stable_id, 0);
		end

		for i = 2:length(rois)
			if params.norm_mode == 0
				rois_event("mod_shard", rois(i).event_frame_id, rois(i).stable_id, fista_fit(rois(i).normalized2, br.normalized2, params), br.stable_id, 0);
			elseif params.norm_mode == 1
				rois_event("mod_shard", rois(i).event_frame_id, rois(i).stable_id, br.total / rois(i).total, br.stable_id, 0);
			elseif params.norm_mode == 2
				rois_event("mod_shard", rois(i).event_frame_id, rois(i).stable_id, br.brightness / rois(i).brightness, br.stable_id, 0);
			end
		end
	end
	success = 1;
end

% Find the ROIs that overlap by the bounding box with the new one.
%
% Args:
% rois - vector of existing ROIs
% nr - the new ROI to compare to
%
% Returns:
% overlap - the indexes of ROIs with overlapping bounding boxes.
function overlap = find_overlap(rois, nr)
	% old ROIs that are overlapping
	overlap = [];

	% order can be reversed to experiment with its effects on merging
	for oi = length(rois):-1:1
	%for oi = 1:length(rois)
		or = rois(oi);

		if or.bbox.col_min > nr.bbox.col_max ...
		|| nr.bbox.col_min > or.bbox.col_max ...
		|| or.bbox.row_min > nr.bbox.row_max ...
		|| nr.bbox.row_min > or.bbox.row_max ...
			continue;
		end
		overlap = [overlap oi];
	end
end

% Remove a ROI at given index from the array.
% The ids of elements get adjusted to match their new positions.
% The ids of elements in new_rois list (the list of ROIs to be processed by
% the combining logic yet) are also updated if they had higher ids.
%
% Args:
% rois - the list of ROIs to remove an element from
% new_rois - the list of ROIs that are copies of those in rois
% idx - index of element in rois to remove
%
% Results:
% rois - the updated list, with the designated element removed, and the
%   ids of the following elements updated to reflect their new position.
% new_rois - the updated list of copied ROIs, where the elements with
%   ids higher than idx get their ids reduced by one, to preserve the
%   correct referencing.
function [rois, new_rois] = remove_roi_at_idx(rois, new_rois, idx)
	rois = [rois(1:(idx-1)) rois((idx+1):end)];
	for i = idx:length(rois)
		rois(i).id = i;
	end
	for i = 1:length(new_rois)
		if new_rois(i).id > idx
			new_rois(i).id = new_rois(i).id - 1;
		end
	end
end

% Add a new ROIs to the list of temporally-far ROIs.
%
% Args:
% rois - list to add to
% nr - new ROI, will have its id field set properly
%
% Returns:
% rois - list with new ROI added
function rois = add_roi_far(rois, nr)
	nr.id = length(rois) + 1;
	if nr.stable_id < 0
		nr.stable_id = rois_new_stable_id();
	end
	LOG("DEBUG ROI "+ string(nr.id)+ " (stable "+ string(nr.stable_id)+ ") added on frame="+ string(nr.frame_id));
	nr.merge_trace = {"+" nr.frame_id nr.id 0 };
	rois = [rois nr];
end

function LOG(msg)
	% disp(msg);
end
