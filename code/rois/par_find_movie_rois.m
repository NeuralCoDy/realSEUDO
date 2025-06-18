% PROTOTYPE, DOESN'T WORK
%
% Find the potential ROIs in a movie fragment, combining them with the previously
% found ROIs.
%
% If parameters request an online computation, clears the rois events and logs
% the new ones.
%
% Example:
%   [rois, rnear] = find_movie_rois([], [], M, 1, -1, rois_params())
%   show_rois(rois, 0)
%
% Example:
%   [rois, rnear] = find_movie_rois([], [], M, 1, 1000, rois_params('online', 1, 'blobify', 1))
%   rois_event("get");
%   show_rois(rois, 0)
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
% movie - frames of the movie as a 3-dimensional array (Y) * (X) * (total_frames)
% frame_id_first - first frame id where we'll be searching for ROIs
% frame_id_last - last frame id where we'll be searching for ROIs, or -1 to the end
% params - parameters produced with roi_params()
%
% Returns:
% rois - the ROIs with changes per the newly analyzed frames, see find_still_rois() for details
% recent_rois - the ROIs found in the last few frames, not combined into the main array yet

function [rois, recent_rois] = par_find_movie_rois(rois, recent_rois, movie, frame_id_first, frame_id_last, params)
	extended_state_version = 4;
	candidate_rois = [];
	if params.extended_state
		if extended_state_version == 4
			initial_rois = rois;
		end
	end

	if ~isempty(params.logfile)
		diary('off');
		diary(params.logfile);
		diary('on');
	end

	if frame_id_last < 0 || frame_id_last > size(movie, 3) - params.avg_frames + 1
		frame_id_last = size(movie, 3) - params.avg_frames + 1;
	end

	if params.online || params.offline
		if params.online && params.offline 
			error("Parameters 'online' and 'offline' must not be set together");
		end
		rois_event("set", params.events);
	end

	% the CPU usage stats are in "jiffies", this is the count of jiffies per second
	jps=get_jps();

	start_time = posixtime(datetime);
	last_time = start_time;
	if params.blobify || params.online
		log_frames = 50;
	else
		log_frames = 500;
	end

	[start_user, start_system] = get_cpustats(jps);
	last_user = start_user;
	last_system = start_system;

	blob_weights = [];

	% The pipeline runs as a parallel computation, with each stage of
	% the pipeline being one of the parallel
	% result of parallel computation, a list of structures with fields:
	% 
	% stage - stage of the pipeline
	% frame_id - frame processed by this stage
	% frame - the structure of frame being processed
	% rois - the temprally-far rois
	% blob_weights - cached weights of the blobs 
	%
	% The pipeline gets primed by a pseudo-result for the previous
	init.stage = 0; % init, a special case
	init.frame_id = frame_id_first;
	init.frame = [];
	init.rois = [];
	init.blob_weights = [];
	par_res = [init];

	while ~isempty(par_res)
		prev_res = par_res;
		par_res = [];
		parfor stage = 1:7
			% Find the result of the previous stage and of the same stage on the last iteration
			last_pst = [];
			last_self = [];
			for ps = 1:length(prev_res)
				if prev_res(ps).stage == stage - 1
					last_pst = prev_res(ps);
				elseif prev_res(ps).stage == stage
					last_self = prev_res(ps);
				end
			end

			frame_id = [];
			if ~isempty(last_pst)
				frame_id = last_pst.frame_id;
			end

			if stage == 1 && ~isempty(last_self)
				frame_id = last_self.frame_id + 1;
			end

			if isempty(frame_id)
				% all done, no more frames
				continue;
			end

			disp(sprintf("SBXXX frame %d stage %d", frame_id, stage));

			% result will be filled here
			my_res = init;
			my_res.stage = stage; 
			my_res.frame_id = frame_id;

			if stage == 1
				if frame_id > frame_id_last
					% all done, no more injecting work
					continue;
				end
				my_res.frame = rois_extract_frame(movie, frame_id, params);
			elseif stage == 2
				rois = []; % TODO a placeholder for now, just blobify
				blob_weights = [];
				if ~isempty(last_self)
					blob_weights = last_self.blob_weights;
				end
				[my_res.frame, rois, my_res.blob_weights] = parse_frame(frame_id, last_pst.frame, rois, blob_weights, params);
			end

			% produce the result
			par_res = [par_res my_res];
		end
	end


	% SBXXXX
	return
	for frame_id = frame_id_first:frame_id_last
		if mod(frame_id, log_frames) == 0
			disp("Processing frame "+ string(frame_id)+ ", found "+ string(length(rois)));
			if frame_id ~= frame_id_first
				prev_time = last_time;
				last_time = posixtime(datetime);

				prev_user = last_user;
				prev_system = last_system;
				[last_user, last_system] = get_cpustats(jps);

				disp(sprintf("Speed average %.3f FPS, recent %.3f FPS, CPU recent u=%.1f%% s=%.1f%% all=%.1f%%, avg u=%.1f%% s=%.1f%% all=%.1f%%", ...
					(frame_id-frame_id_first) / (last_time-start_time), ...
					log_frames / (last_time-prev_time), ...
					(last_user-prev_user) * 100 / (last_time-prev_time), ...
					(last_system-prev_system) * 100 / (last_time-prev_time), ...
					((last_user+last_system) - (prev_user+prev_system)) * 100 / (last_time-prev_time), ...
					(last_user-start_user) * 100 / (last_time-start_time), ...
					(last_system-start_system) * 100 / (last_time-start_time), ...
					((last_user+last_system) - (start_user+start_system)) * 100 / (last_time-start_time)));

			end
		end

		% see what ROIs have been unchanged long enough, and merge them into the main set
		if ~params.offline
			[recent_rois, done_rois] = find_done_rois(recent_rois, frame_id, params.recent_frames);
			rois = combine_rois(rois, done_rois, "far", params);
		end

		% update the recent set from the new frame
		frame = rois_extract_frame(movie, frame_id, params);
		[frame, rois, blob_weights] = parse_frame(frame_id, frame, rois, blob_weights, params);

		if ~params.offline
			new_rois = find_still_rois_in(frame, params);
			recent_rois = combine_rois(recent_rois, new_rois, "near", params);

			% do a detection of events on the tentative ROIs
			[recent_rois] = detect_early(frame_id, frame, rois, recent_rois, blob_weights, params);
		end
	end

	if params.flush && ~params.offline
		% flush out into combining all the near ROIs
		[recent_rois, done_rois] = find_done_rois(recent_rois, frame_id_last + 1, 0);
		rois = combine_rois(rois, done_rois, "far", params);
	end

	last_time = posixtime(datetime);
	[last_user, last_system] = get_cpustats(jps);

	disp(sprintf("Speed average %.3f FPS, CPU average u=%.1f%% s=%.1f%% all=%.1f%%", ...
		(frame_id-frame_id_first) / (last_time-start_time), ...
		(last_user-start_user) * 100 / (last_time-start_time), ...
		(last_system-start_system) * 100 / (last_time-start_time), ...
		((last_user+last_system) - (start_user+start_system)) * 100 / (last_time-start_time)));

	if params.online
		disp("Extracting activation traces.");
		for i = 1:length(rois)
			rois(i).trace = nan; % wipe out the previous trace if any
			rois(i).trace = rois_trace(rois(i));
		end
	end

	if ~isempty(params.logfile)
		diary('off');
	end
end

% Find what ROIs have become not recent any more, and split them out.
%
% Args:
% recent_rois - the recent ROI set
% frame_id - id of the next frame
% max_recent_frames - duration in frames for the ROI to stay recent
%
% Returns:
% new_recent_rois - the recent ROI set with the ROIs that became non-recent removed
% done_rois - the ROIs that became non-recent
function [new_recent_rois, done_rois] = find_done_rois(recent_rois, frame_id, max_recent_frames)
	new_recent_rois = [];
	done_rois = [];
	for i = 1:length(recent_rois)
		if recent_rois(i).frame_id < frame_id - max_recent_frames
			done_rois = [done_rois recent_rois(i)];
		else
			new_recent_rois = [new_recent_rois recent_rois(i)];
		end
	end
end


% Parse the data in a frame, do the online event detection and blobification
% as requested by the parameters.
function [frame, rois, blob_weights] = parse_frame(frame_id, frame, rois, blob_weights, params)
	if params.online || params.offline
		rois_shapes = normalized2_shapes(frame.ht, frame.wd, rois);
		if ~isempty(rois)
			rois_weights = [rois(:).last_weight];

			if isempty(blob_weights)
				blob_weights = zeros(1, frame.ht * frame.wd);
			end
		else
			rois_weights = [];
		end

		if params.blobify
			blob = params.blobify_blob;
			blob_spacing = params.blob_spacing;
		else
			blob = [];
			blob_spacing = frame.ht*10;
		end

		[weights, steps] = seudo_native(frame.pixels, blob, blob_spacing, ...
			rois_shapes, [rois_weights, blob_weights], [], params.fit_tol, params.fit_max_steps, ...
			params.fit_l_mode, params.fit_stop_mode, false, params.fit_parallel);

		for i = 1:length(rois)
			w = weights(i);
			rois(i).last_weight = w;

			% Uncomment to get the unnormalized fitting.
			% divide, because weight is inversely proportional to the magnitude of ROI
			% w = w / rois(i).normalize_coeff2;

			if w >= params.min_detect_fit
				rois_event("detect", frame_id, rois(i).stable_id, w, 0, rois(i).normalize_coeff2);
				rois(i).event_frame_id = frame_id;
			end
		end
		blob_weights = reshape(weights(1+length(rois):end), 1, []);

		if params.blobify && ~params.offline
			% rebuild the smoothed frame from blobs and ROIs
			frame.pixels = convn(reshape(blob_weights, frame.ht, frame.wd), params.blobify_blob, 'same') ...
				+ detected_pixels(frame.ht, frame.wd, rois);
		end
	else
		if params.blobify
			[blob_weights, steps] = seudo_native(frame.pixels, params.blobify_blob, params.blob_spacing, ...
				[], blob_weights, [], params.fit_tol, params.fit_max_steps, ...
				params.fit_l_mode, params.fit_stop_mode, false, params.fit_parallel);
			frame.pixels = convn(reshape(blob_weights, frame.ht, frame.wd), params.blobify_blob, 'same');
		end
	end
end

% do a detection of events on the tentative ROIs
function [recent_rois] = detect_early(frame_id, frame, rois, recent_rois, blob_weights, params)
	if params.online && ~isempty(recent_rois)
		% try to report the brightness of the tentative ROIs

		% subtract the known confirmed ROIs from the frame;
		% this is cheaper and gives them more priority than trying to
		% fit both kinds of ROIs at the same time
		if params.blobify
			% just rebuild the frame from blobs without detected ROIs
			frame.pixels = convn(reshape(blob_weights, frame.ht, frame.wd), params.blobify_blob, 'same');
		elseif ~isempty(rois)
			frame.pixels = frame.pixels - detected_pixels(frame.ht, frame.wd, rois);
		end

		% prepare the arguments for detection
		rois_shapes = normalized2_shapes(frame.ht, frame.wd, recent_rois);
		rois_weights = [recent_rois(:).last_weight];

		if params.blobify
			blob = params.blobify_blob;
			blob_spacing = params.blob_spacing;
		else
			blob = [];
			blob_spacing = frame.ht*10;
		end

		[weights, steps] = seudo_native(frame.pixels, blob, blob_spacing, ...
			rois_shapes, [rois_weights, blob_weights], [], params.fit_tol, params.fit_max_steps, ...
			params.fit_l_mode, params.fit_stop_mode, false, params.fit_parallel);

		for i = 1:length(recent_rois)
			w = weights(i);
			recent_rois(i).last_weight = w;

			if recent_rois(i).stable_id < 0 && w >= params.min_detect_early_fit
				recent_rois(i).stable_id = rois_new_stable_id();
			end

			if recent_rois(i).stable_id >= 0 && w >= params.min_detect_fit
				rois_event("detect_early", frame_id, recent_rois(i).stable_id, w, 0, recent_rois(i).normalize_coeff2);
				recent_rois(i).event_frame_id = frame_id;
			end
		end
	end
end

% Extract the normalized2 shapes from ROIs, reshaped for passing to
% seudo_native().
%
% Args:
% ht - frame height
% wd - frame width
% rois - the ROIs array
function rois_shapes = normalized2_shapes(ht, wd, rois)
	if ~isempty(rois)
		% The meaning is this 
		%   rois_shapes = [];
		%   for i = 1:length(rois)
		%     rois_shapes = [rois_shapes rois(i).normalized2];
		%   end
		%   rois_shapes = reshape(rois_shapes, ht * wd, []);
		rois_shapes = reshape([rois(:).normalized2], ht * wd, []);
	else
		rois_shapes = [];
	end
end

% Extract the pixels of all the ROIs detected in the last frame.
%
% Args:
% ht - frame height
% wd - frame width
% rois - the ROIs array
function rois_pixels = detected_pixels(ht, wd, rois)
	% The meaning is this but expressed more Matlab-efficient:
	%   for i = 1:length(rois)
	%     rois_pixels = rois_pixels + (rois(i).normalized2 * rois(i).last_weight);
	%   end
	%
	if ~isempty(rois)
		rois_pixels = sum(reshape( ...
			reshape([rois(:).normalized2], ht * wd, []) * [rois(:).last_weight]', ...
			ht, wd, []), 3);
	else
		rois_pixels = zeros(ht, wd);
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
% user - process CPU usage in user mode
% system - process CPU usage in system mode
function [user, system] = get_cpustats(jps)
	raw = fileread("/proc/self/stat");
	fields=string(split(raw));
	user = double(fields(14)) / jps;
	system = double(fields(15)) / jps;
end
