% Find the least-squares fit of one image into another one.
%
% Args:
% what - what to fit, the exact dimensions of this image don't matter as long
%   as the number of the pixels is the same as in the target, it will be
%   reshaped anyway
% target - where to fit into, the effect of the exact dimenstions depends
%   on what internal implementation is selected: for exact least-squares
%   it doesn't matter, for FISTA approximation the higher the number of
%   rows the greater the possible parallelism.
% params - rois_params() controlling parameters of parallelism
%
% Returns:
% score - the fit score
function score = fista_fit(what, target, params)
	if 1
		% the exact least-squares in Matlab, tends to be a little faster on bigger machines

		% it complains periodically about singular matrices
		warning('off','MATLAB:singularMatrix')

		x = reshape(what, [], 1);
		y = reshape(target, [], 1);
		score = max(0, inv(x' * x) * x' * y);

		warning('on','MATLAB:singularMatrix')
	else
		% the FISTA approximation
		[score, steps] = seudo_native(target, [0], 100000, reshape(what, [], 1), [], [], 0.001, 100, 2, 0, false, params.fit_parallel);
		score = score(1);
	end
end

