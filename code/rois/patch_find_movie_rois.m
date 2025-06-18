% Find the ROIs by patches.
% It's somewhat simple-minded, not caring about the ROIs that get split by
% the patch boundaries.
%
% Arguments are mostly based on find_movie_rois() with extra arg and a little
% simplification.
%
% Args:
% rois - vector of previously found ROIs, see find_still_rois() for details. If the
%   online mode is selected in the parameters, the ROIs will have the trace field
%   filled in, and also the full events will be in rois_event() buffer. Note that
%   in the incremental online computation the events from the previous incremental
%   run won't be automatically transferred, they have to be copied through
%   params.events.
% recent_rois - vector of the ROIs found in the recently preceding frames;
%   the temporal combining between ROIs is done differently between the recent
%   and older frames
% split_x - number of patches to split into by X dimension
% split_y - number of patches to split into by Y dimension
% movie - frames of the movie as a 3-dimensional array (Y) * (X) * (total_frames)
% params - parameters produced with roi_params()
%
% Returns:
% o_rois - the ROIs with changes per the newly analyzed frames, see find_still_rois() for details
% o_recent_rois - the ROIs found in the last few frames, not combined into the main array yet

function [o_rois, o_recent_rois] = patch_find_movie_rois(rois, recent_rois, split_x, split_y, movie, params)
	movie_ht = size(movie, 1);
	movie_wd = size(movie, 2);

	patch_ht = ceil(movie_ht / split_y);
	patch_wd = ceil(movie_wd / split_x);

	patches = [];

	if ~isempty(params.logfile)
		diary('off');
		diary(params.logfile);
		diary('on');
	end

	for ix = 0:split_x-1
		for iy = 0:split_y-1
			p = params;
			p.logfile = [];
			p.movie_ht = movie_ht;
			p.movie_wd = movie_wd;
			p.patch_offset_y = iy * patch_ht;
			p.patch_offset_x = ix * patch_wd;
			p.patch_id = length(patches) + 1;

			from_y = 1 + p.patch_offset_y;
			from_x = 1 + p.patch_offset_x;
			to_y = patch_ht + p.patch_offset_y;
			to_x = patch_wd + p.patch_offset_x;

			if to_y > movie_ht
				to_y = movie_ht;
			end
			if to_x > movie_wd
				to_x = movie_wd;
			end

			args = struct('params', p, 'movie', movie(from_y:to_y, from_x:to_x, :));
			patches = [patches args];
		end
	end

	o_rois = [];
	o_recent_rois = [];

	% the CPU usage stats are in "jiffies", this is the count of jiffies per second
	jps=get_jps();

	if ~params.patch_parallel
		start_time = posixtime(datetime);
		[start_user, start_system] = get_xcpustats(jps);

		% serial execution
		for i = 1:length(patches)
			[rx, rrx] = find_movie_rois(rois, recent_rois, patches(i).movie, 1, -1, patches(i).params);
			o_rois = [o_rois rx];
			o_recent_rois = [o_recent_rois rrx];
		end

		last_time = posixtime(datetime);
		[last_user, last_system] = get_xcpustats(jps);
	else
		% parallel execution

		nproc = params.patch_nproc;
		if nproc <= 0
			nproc = split_x * split_y;
		end

		pool = gcp('nocreate');
		if ~isempty(pool) 
			% stop the existing pool, to count the usage stats from scratch
			delete(pool);
			pool = gcp('nocreate');
		end
		if isempty(pool)
			parpool('processes', nproc);
		end

		% pool start may take a long time, so remember the new start time
		start_time = posixtime(datetime);
		[start_user, start_system] = get_xcpustats(jps);

		parfor i = 1:length(patches)
			[rx, rrx] = find_movie_rois(rois, recent_rois, patches(i).movie, 1, -1, patches(i).params);
			o_rois = [o_rois rx];
			o_recent_rois = [o_recent_rois rrx];
		end

		if params.patch_merge
			disp("Merging the ROIs by patch boundaries");
			o_rois = rois_merge_patch_all(o_rois, split_x, split_y, params);
		end

		last_time = posixtime(datetime);

		% stop the pool, to collect its CPU usage stats
		delete(gcp('nocreate'));

		[last_user, last_system] = get_xcpustats(jps);
	end
	
	disp(sprintf("Total speed average %.3f FPS, CPU u=%.1f%% s=%.1f%% all=%.1f%%", ...
		size(movie, 3) / (last_time-start_time), ...
		(last_user-start_user) * 100 / (last_time-start_time), ...
		(last_system-start_system) * 100 / (last_time-start_time), ...
		((last_user+last_system) - (start_user+start_system)) * 100 / (last_time-start_time)));

	if ~isempty(params.logfile)
		diary('off');
	end
end

% Read the jiffies-per-second translation coefficient from Linux.
% It is used to convert the CPU usage stats from jiffies to seconds.
function jps = get_jps()
	[cmdstat, cmdout] = system("getconf CLK_TCK");
	if cmdstat ~= 0
		error("Could not get jiffies-per-second");
	end
	jps_str=string(split(cmdout));
	jps=double(jps_str(1));
end

% Get the CPu stats of the process.
%
% Args:
% jps - jiffies per second translation coefficient
%
% Returns:
% user - process and waited children CPU usage in user mode
% system - process and waited children CPU usage in system mode
function [user, system] = get_xcpustats(jps)
	raw = fileread("/proc/self/stat");
	fields=string(split(raw));
	user = double(fields(14)) + double(fields(16))/ jps;
	system = double(fields(15)) + double(fields(17))/ jps;
end
