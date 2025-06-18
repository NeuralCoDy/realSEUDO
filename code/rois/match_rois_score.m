% Compute the score for a match of two ROIs produced by different
% recognition algorithms (or humans) that takes both the shape and
% time trace into account.
%
% Args:
% ra - the ROI from first source
% tracea - time trace of the ROI from the first source, 1*NF
% rb - the ROI from second source
% traceb - time trace of the ROI from the second source, 1*NF
% params - rois_params() defining the constants for matching
%
% Parameters:
% maxtracea - pre-computed max value from tracea
% avgtracea - pre-computed average value from tracea
% maxtraceb - pre-computed max value from traceb
% avgtraceb - pre-computed average value from traceb
function score = match_rois_score(ra, tracea, rb, traceb, params, varargin)
	p = inputParser;
	p.addParameter('maxtracea', []);
	p.addParameter('avgtracea', []);
	p.addParameter('maxtraceb', []);
	p.addParameter('avgtraceb', []);
	parse(p,varargin{:});
	maxtracea = p.Results.maxtracea;
	avgtracea = p.Results.avgtracea;
	maxtraceb = p.Results.maxtraceb;
	avgtraceb = p.Results.avgtraceb;

	% equalize the trace lengths by appending 0s
	if length(tracea) < length(traceb)
		tracea = [tracea zeros(1, length(traceb) - length(tracea))];
	elseif length(traceb) < length(tracea)
		traceb = [traceb zeros(1, length(tracea) - length(traceb))];
	end

	% cut the negatives
	tracea = tracea .* (tracea > 0);
	traceb = traceb .* (traceb > 0);

	% normalize the trace brightness
	% use the 90% percentile of the values above average as the estimation of brightness,
	% and bring it to be 1

	if isempty(maxtracea)
		maxtracea = max(tracea);
	end
	if isempty(avgtracea)
		avgtracea = mean(tracea);
	end
	pcoefa = quantile(tracea(find(tracea > avgtracea)), 0.9);

	if isempty(maxtraceb)
		maxtraceb = max(traceb);
	end
	if isempty(avgtraceb)
		avgtraceb = mean(traceb);
	end
	pcoefb = quantile(traceb(find(traceb > avgtraceb)), 0.9);

	% even if a trace is empty, don't give up, try to make a score
	% based purely on shape
	if pcoefa > 0 && pcoefb > 0
		tracea = tracea / pcoefa;
		traceb = traceb / pcoefb;
	end

	% normalize the cell brightness by bringing the 90% percentile to be 1

	coefa = ra.sorted(floor(length(ra.sorted) * 0.9));
	coefb = rb.sorted(floor(length(rb.sorted) * 0.9));

	if coefa == 0 || coefb == 0
		score = 0;
		return;
	end

	rawa = ra.raw / coefa;
	rawb = rb.raw / coefb;

	score = match_score(rawa, ra.mask, tracea, rawb, rb.mask, traceb, params);
end
