% Merge the ROIs that are adjacent to each other along all the patch lines.
% 
% Args:
% rois - the set of ROIS that has been processed by patches (already after
%  bringing to the common dimensions with rois_from_patch())
% split_x - number of patches that were split into by X dimension
% split_y - number of patches that were split into by Y dimension
%
% Returns:
% rois - merged ROIs, the trace will be combined, adjusting for
%   brightness and normalization
function rois = rois_merge_patch_all(rois, split_x, split_y, params)
	if isempty(rois)
		return
	end

	movie_ht = size(rois(1).raw, 1);
	movie_wd = size(rois(1).raw, 2);

	patch_ht = ceil(movie_ht / split_y);
	patch_wd = ceil(movie_wd / split_x);

	for x = patch_wd:patch_wd:movie_wd
		rois = rois_merge_patch_line(rois, "x", x, params);
	end

	for y = patch_ht:patch_ht:movie_ht
		rois = rois_merge_patch_line(rois, "y", y, params);
	end
end

