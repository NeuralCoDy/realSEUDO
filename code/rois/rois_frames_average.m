% Produce the averaging across multiple frames
%
% Args:
% frames - The original frame array X*Y*N
% count - number of frames to average across
%
% Returns:
% avframes - averaged frames, will have (N - (count-1)) frames
function avframes = rois_frames_average(frames, count)
		% distance between the first and last frame
		dist = count - 1;
		avframes = frames(:, :, 1:end-dist);
		for i=1:dist
			avframes = avframes + frames(:, :, 1+i:end-dist+i);
		end

		avframse = avframes ./ count;
end
