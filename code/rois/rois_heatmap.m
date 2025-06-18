% Produce a "heatmap" summarizing the bright areas across
% all the frames. To reduce noise, the frames are first passed
% through a running average of 5 frames.
%
% Args:
% frames - frames to summarize
%
% Results:
% map - the heat map
function map = rois_heatmap(frames)
	nf = size(frames, 3);
	if nf < 5
		error("Need at least 5 frames for averaging");
	end

	% average over 5 frames window, and find the maximum
	if 0
		% this version uses A LOT of memory
		map = max( ...
			frames(:,:,1:end-4) ...
			+ frames(:,:,2:end-3) ...
			+ frames(:,:,3:end-2) ...
			+ frames(:,:,4:end-1) ...
			+ frames(:,:,5:end) ...
			, [], 3) ./ 5;
	else
		map = zeros(size(frames, 1), size(frames, 2));
		for i = 1:nf+1-5
			f = sum(frames(:, :, i:i+4), 3);
			map = max(map, f);
		end

		map = map ./ 5;
	end

end
