function [groupa, groupb, ngrps] = group_rois_by_score(score)
	na = size(score, 1);
	nb = size(score, 2);

	% group ids where the rois are assigned
	groupa = zeros(1, na);
	groupb = zeros(1, nb);
	grpid = 0;

	% the best values and indexes by rows and columns
	[besta, idxa] = max(score, [], 2);
	[bestb, idxb] = max(score, [], 1);

	for a = 1:na
		% start a new group
		grpid = grpid + 1;
		% effective group id
		egrp = grpid;
		
		% take the rois from A
		groupa(a) = egrp;

		% take the best match from B
		if besta(a) > 0
			b = idxa(a);
			if groupb(b)
				% the best match is already claimed, because its
				% best match doesn't reciprocate, so instead add the
				% current ROI to that group
				egrp = groupb(b);
				groupa(a) = egrp;
			else
				groupb(b) = egrp;
			end
		end

		% take the other matches from B for thish this A is the best match
		for b = 1:nb
			if bestb(b) > 0 && idxb(b) == a && ~groupb(b)
				groupb(b) = egrp;
			end
		end
	end

	ngrps = grpid;
end
