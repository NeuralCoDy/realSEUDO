% Merge the ROIs that are adjacent to each other along the patch line.
% 
% Args:
% rois - the set of ROIS that has been processed by patches (already after
%  bringing to the common dimensions with rois_from_patch())
% dim - dimension of patch boundary, "x" or "y"
% boundary - the boundary coordinate value, the merging will happen between
%  boundary and (boundary+1)
%
% Returns:
% rois - merged ROIs, the trace will be combined, adjusting for
%   brightness and normalization
function rois = rois_merge_patch_line(rois, dim, boundary, params)
	if dim ~= "x" && dim ~= "y"
		error("The dim argument must be 'x' or 'y', got '" + string(dim) + "'");
	end

	if isempty(rois)
		return
	end

	if 0
		brightness = ones(1, length(rois));
		for i = 1:length(rois)
			% estimate the brightness as going one standard deviation up from mean
			brightness(i) = mean(rois(i).sorted) + std(rois(i).sorted);
		end
	end

	% This evens out the trace sizes from all the sources,
	% so that they can be placed into a single table.
	all_trace = rois_trace(rois);

	% the set A will be ending at the boundary
	seta = [];
	% its boundary lines
	bnda = [];
	% its masks of the boundary line
	maska = [];
	% its activation traces
	tracea = [];
	for i = 1:length(rois)
		if dim == "x"
			if rois(i).bbox.col_max ~= boundary
				continue;
			end
			seta = [seta i];
			bnda = [bnda; rois(i).raw(:, boundary)' ];
			maska = [maska; rois(i).mask(:, boundary)' ];
			tracea = [tracea; all_trace(i, :) / rois(i).normalize_coeff2 ];
		else
			if rois(i).bbox.row_max ~= boundary
				continue;
			end
			seta = [seta i];
			bnda = [bnda; rois(i).raw(boundary, :) ];
			maska = [maska; rois(i).mask(boundary, :) ];
			tracea = [tracea; all_trace(i, :) / rois(i).normalize_coeff2 ];
		end
	end

	% the set B will be starting at the boundary
	setb = [];
	% its boundary lines
	bndb = [];
	% its masks of the boundary line
	maskb = [];
	% its activation traces
	traceb = [];
	for i = 1:length(rois)
		if dim == "x"
			if rois(i).bbox.col_min ~= boundary+1
				continue;
			end
			setb = [setb i];
			bndb = [bndb; rois(i).raw(:, boundary+1)' ];
			maskb = [maskb; rois(i).mask(:, boundary+1)' ];
			traceb = [traceb; all_trace(i, :) / rois(i).normalize_coeff2 ];
		else
			if rois(i).bbox.row_min ~= boundary+1
				continue;
			end
			setb = [setb i];
			bndb = [bndb; rois(i).raw(boundary+1, :) ];
			maskb = [maskb; rois(i).mask(boundary+1, :) ];
			traceb = [traceb; all_trace(i, :) / rois(i).normalize_coeff2 ];
		end
	end

	if length(seta) == 0 || length(setb) == 0
		% no candidates for merging
		return
	end
	
	% similar to match_rois_all()
	score = zeros(length(seta), length(setb));
	for a = 1:length(seta)
		for b = 1:length(setb)
			score(a, b) = match_score(bnda(a, :), maska(a, :), tracea(a, :), bndb(b, :), maskb(b, :), traceb(b, :), params);
		end
	end

	% throw away the weak matches
	score = score .* (score >= 2);

	% this will mark, which ROIs have been merged
	merged = zeros(1, length(rois));
	% the output ROIs after change will be collected here
	orois = [];

	% similar to show_matched_trace.m
	[groupa, groupb, ngrps] = group_rois_by_score(score);

	for egrp = 1:ngrps
		% these lists are of indexes in seta/setb
		lista = find(groupa == egrp);
		listb = find(groupb == egrp);
		grpsz = length(lista) + length(listb);

		% if no match, don't try to group
		if grpsz < 2
			continue;
		end

		% lista is guaranteed to have at least 1 entry, use it as baseline
		a = lista(1);
		i = seta(a);
		merged(i) = 1;
		r = rois(i);
		r.id = length(orois) + 1;
		r.stable_id = r.id;
		% the partial traces will be weighed by pixel count
		trace = tracea(a, :) * r.count;
		traceweight = r.count;

		%disp("Starting group " + string(egrp) + " with roi " + string(seta(lista(1))) + " from side A");

		% Find and merge the ROIs from A
		for a = lista(2:end)
			i = seta(a);
			merged(i) = 1;
			%disp("Extending group " + string(egrp) + " with roi " + string(i) + " from side A");

			r.frame = max(r.frame, rois(i).frame);
			r.mask = max(r.mask, rois(i).mask);
			r.raw = max(r.raw, rois(i).raw);
			trace = trace + tracea(a, :) * r.count;
			traceweight = traceweight + r.count;
		end

		% Find and merge the ROIs from B
		for b = listb
			i = setb(b);
			merged(i) = 1;
			%disp("Extending group " + string(egrp) + " with roi " + string(i) + " from side B");

			r.frame = max(r.frame, rois(i).frame);
			r.mask = max(r.mask, rois(i).mask);
			r.raw = max(r.raw, rois(i).raw);
			trace = trace + traceb(b, :) * r.count;
			traceweight = traceweight + r.count;
		end

		r.count = sum(r.mask, "all");
		r = roi_postcreate(r);

		% de-scale the trace and normalize it
		r.trace = trace * (r.normalize_coeff2 / traceweight);

		orois = [orois r];
	end

	if ~isempty(orois)
		% made some merges, now need to move over the unmerged ROIs
		for i = 1:length(rois)
			if ~merged(i)
				r = rois(i);
				r.id = length(orois) + 1;
				r.stable_id = r.id;
				orois = [orois r];
			else
				%disp("ROI " + string(i) + " is already merged");
			end
		end

		rois = orois;
	end
end
