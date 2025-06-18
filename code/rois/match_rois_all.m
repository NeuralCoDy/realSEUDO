function score = match_rois_all(ra, rb, params, varargin)
	p = inputParser;
	p.addParameter('filter', 0);
	p.addParameter('minval', 0);
	parse(p,varargin{:});
	filter = p.Results.filter;
	minval = p.Results.minval;

	score = zeros(length(ra), length(rb));

	% compute the trace stats once
	maxtracea = zeros(1, length(ra));
	avgtracea = zeros(1, length(ra));
	for a = 1:length(ra)
		maxtracea(a) = max(ra(a).trace);
		avgtracea(a) = mean(ra(a).trace);
	end

	maxtraceb = zeros(1, length(rb));
	avgtraceb = zeros(1, length(rb));
	for b = 1:length(rb)
		maxtraceb(b) = max(rb(b).trace);
		avgtraceb(b) = mean(rb(b).trace);
	end

	for a = 1:length(ra)
		for b = 1:length(rb)
			score(a, b) = match_rois_score(ra(a), ra(a).trace, rb(b), rb(b).trace, params, ...
				'maxtracea', maxtracea(a), 'avgtracea', avgtracea(a), ...
				'maxtraceb', maxtraceb(b), 'avgtraceb', avgtraceb(b));
		end
	end

	if filter
		score = match_rois_filter(score, 'minval', minval);
	end
end
