% Filter the score matrix for the match of two sets of ROIs.
% Removes the low-quality matches when higher-quality matches
% are available.
%
% Args:
% score - the raw score matrix
%
% Options:
% minval - the minimal score to accept at all
function oscore = match_rois_filter(score, varargin)
	p = inputParser;
	p.addParameter('minval', 0);
	parse(p,varargin{:});
	minval = p.Results.minval;

	[na, nb] = size(score);

	% the best values and indexes by rows and columns
	[besta, idxa] = max(score, [], 2);
	[bestb, idxb] = max(score, [], 1);

	oscore = zeros(size(score));

	% Enter the highest score for each ROI
	% but throw away those that are very weak and way below
	% other side's best match.
	for a = 1:na
		if besta(a) >= minval && (besta(a) >= 1 || bestb(idxa(a)) < 1)
			oscore(a, idxa(a)) = besta(a);
		end
	end
	for b = 1:nb
		if bestb(b) >= minval && (bestb(b) >= 1 || besta(idxb(b)) < 1)
			oscore(idxb(b), b) = bestb(b);
		end
	end

end
