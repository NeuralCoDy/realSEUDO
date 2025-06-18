% Estimate scale of the trace for one ROI,
% and rescale it to the "standard scale"
% that moves the trace to above 0 amd puts the 90% percentile
% of values above average at 1.
%
% Args:
% trace - traces (Nrois, Nframes) to rescale
function trace = rescale_trace(trace)
	offset = min(trace, [], "all");
	if offset > 0
		% only the negative values are considered an offset,
		% to move the trace above 0
		offset = 0;
	end

	% use the 90% percentile of the values above average as the estimation of brightness
	avgtrace = mean(trace, 2);
	scale = quantile(trace(find(trace > avgtrace)), 0.9);
	scale = scale - offset;

	if scale ~= 0
		trace = (trace - offset) ./ scale;
	end
end
